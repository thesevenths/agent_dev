"""ARR-07: SciNet++ loop with ① LLM-proposed structure/params and ③ LLM finish.

No human-authored candidate menu, no designer-chosen parameter values, no
GoalGate deciding finish. The LLM decides both; the harness only:
  - validates proposed keys/ranges against the REAL SciNet config schema,
  - runs real train+evaluate,
  - records LLM decisions + raw metrics as auditable evidence.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import config  # noqa: F401  (loads .env via load_dotenv)
from research_runtime.llm_client import llm_config, make_generate

from . import core, scinet_backend
from ..generic.llm_autonomy import LLMResearcher
from ..generic.research_contract import (
    MetricGoal, ResearchContract, sanity_check, SearchMemory, paired_seed_verdict,
    freeze_contract, load_contract, load_dataset_scale)

# REAL SciNet++ config schema with TYPE + SCALE validation.
# LLM chooses values freely WITHIN these constraints (not from a fixed menu),
# but cannot submit a structure the engine cannot build (e.g. int for hidden
# layers -> SciNet's [*hidden] crashes).
SCHEMA = {
    "train.lr": (0.00001, 0.1),
    "jepa.pred_weight": (0.0, 100.0),
    "jepa.ema_momentum": (0.5, 0.9999),
    "jepa.vicreg_var_weight": (0.0, 100.0),
    "jepa.vicreg_cov_weight": (0.0, 100.0),
    "model.latent_dim": (2, 32),          # tiny smoke dataset (64 samples)
    "model.predictor_hidden": "list_int", # MUST be a list of ints (SciNet build_mlp)
}
OBJECTIVE = ("LeJEPA-style study: improve held-out latent prediction by tuning "
             "regularisation (VICReg variance/covariance) AND model structure. "
             "Minimize latent_mse@1 AND maximize cosine@1 jointly.")

@dataclass
class AutonomousResult:
    task_id: str
    finished: bool
    iterations: int
    final_improvement: float | None
    rounds: list[dict[str, Any]] = field(default_factory=list)
    llm_calls: int = 0
    engine_runs: int = 0
    budget_exhausted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "finished": self.finished,
                "iterations": self.iterations,
                "final_improvement": self.final_improvement,
                "llm_calls": self.llm_calls, "engine_runs": self.engine_runs,
                "budget_exhausted": self.budget_exhausted,
                "rounds": self.rounds}


def run_autonomous(*, task_id: str, root: Path, seed: int = 42,
                   max_iterations: int = 3, target: float = 0.10,
                   protocol_hash: str = "") -> AutonomousResult:
    root = Path(root).resolve()
    cfg = llm_config()
    if not cfg.get("api_key"):
        raise RuntimeError("LLM not configured")
    researcher = LLMResearcher(make_generate(),
                               root / "raw_data" / "autonomy" / core.sha256_json(task_id)[:10])
    memory = SearchMemory(root / "raw_data" / "autonomy" / "search_memory.json")
    res = AutonomousResult(task_id=task_id, finished=False, iterations=0,
                           final_improvement=None)
    # Contract comes from the CAMPAIGN FILE (frozen before run), not Python code.
    # Different domains/tests supply their own contract.json -> same harness code.
    if not (root / "contract.json").exists():
        freeze_contract(root, ResearchContract(
            goals=(MetricGoal("latent_mse@1", "min", 0.10),
                   MetricGoal("cosine@1", "max", 0.0)),
            budget=max_iterations, max_params_per_train_sample=8.0))
        (root / "dataset_scale.json").write_text(
            json.dumps({"n_train_samples": 64}), encoding="utf-8")
    contract = load_contract(root)
    n_train = load_dataset_scale(root)
    prior_path = root / "prior_evidence.json"
    prior_evidence = (json.loads(prior_path.read_text(encoding="utf-8"))
                      if prior_path.is_file() else {})

    base = scinet_backend.run_scinet_experiment(
        experiment_id=f"auto_{core.sha256_json(task_id)[:8]}_base",
        task_id=task_id, seed=seed, overrides={}, root=root,
        protocol_hash=protocol_hash)
    if not base.get("ok"):
        res.rounds.append({"error": base.get("error"), "phase": "baseline"})
        return res
    res.engine_runs += 1
    baseline = {k: float(v) for k, v in base["metrics"].items()}
    bm = baseline.get(scinet_backend.DEFAULT_METRIC, 0.0)
    latest = baseline
    imp = 0.0
    history: list[dict] = []
    seed_pairs: list[tuple[tuple[float, float], tuple[float, float]]] = []

    for it in range(max_iterations):
        # ① LLM proposes structure/params, WITH frozen goal + search memory in prompt.
        p = researcher.propose_experiment(
            objective=OBJECTIVE, allowed=SCHEMA,
            baseline_metrics=baseline, history=history,
            search_hint=memory.direction_hint(metric=scinet_backend.DEFAULT_METRIC),
            contract={"goals": [g.to_dict() for g in contract.goals],
                      "rule": contract.evaluate(baseline, latest)["rule"]},
            max_changes=2,
            prior_evidence=prior_evidence)
        if not p.valid:
            res.rounds.append({"iteration": it, "phase": "proposal",
                               "valid": False, "reason": p.rejection_reason,
                               "raw": p.raw_response[:500]})
            history.append({"iteration": it, "proposal_valid": False,
                            "reason": p.rejection_reason})
            continue
        # ④ STRUCTURAL SANITY in code: reject absurd structures for a 64-sample set.
        sane = sanity_check(p.overrides, n_train_samples=n_train,
                            ratio_limit=contract.max_params_per_train_sample)
        if not sane["ok"]:
            res.rounds.append({"iteration": it, "phase": "sanity",
                               "overrides": p.overrides, "rejected": True,
                               "reason": sane["reason"]})
            history.append({"iteration": it, "overrides": p.overrides,
                            "sanity_rejected": True, "reason": sane["reason"]})
            continue
        if memory.seen(p.overrides):
            res.rounds.append({"iteration": it, "phase": "search_memory",
                               "overrides": p.overrides, "duplicate": True,
                               "reason": "configuration already tried; LLM should propose a new direction"})
            history.append({"iteration": it, "overrides": p.overrides,
                            "duplicate": True})
            continue
        exp_id = f"auto_{core.sha256_json(task_id)[:8]}_it{it}"
        rec = scinet_backend.run_scinet_experiment(
            experiment_id=exp_id, task_id=task_id, seed=seed,
            overrides=p.overrides, root=root, protocol_hash=protocol_hash)
        res.engine_runs += 1
        engine_ok = bool(rec.get("ok")) and bool(rec.get("metrics"))
        measured = {k: float(v) for k, v in (rec.get("metrics") or {}).items()} \
            if engine_ok else {}
        em = measured.get(scinet_backend.DEFAULT_METRIC)
        imp = None if em is None else ((bm - em) / abs(bm) if abs(bm) > 1e-12 else 0.0)
        verdict = contract.evaluate(baseline, measured or latest)
        # ⑤ Paired multi-seed evidence (record; CI used only when >=2 seeds).
        if engine_ok:
            b_pair = (baseline.get(scinet_backend.DEFAULT_METRIC, 0.0),
                      baseline.get("cosine@1", 0.0))
            a_pair = (measured.get(scinet_backend.DEFAULT_METRIC, 0.0),
                      measured.get("cosine@1", 0.0))
            seed_pairs.append((b_pair, a_pair))
        ci = paired_seed_verdict([x[0] for x in seed_pairs],
                                 [x[1] for x in seed_pairs]) if len(seed_pairs) >= 2 \
            else {"note": "single seed - no paired CI yet"}
        # ③ LLM decides finish, fed the FROZEN quantified verdict.
        dec = researcher.decide_finish(
            objective=OBJECTIVE, target=target, baseline=baseline,
            latest=measured or latest, improvement=imp,
            iteration=it, max_iterations=max_iterations,
            contract_verdict={**verdict, "paired_seed_ci": ci})
        memory.record(iteration=it, overrides=p.overrides, metrics=measured,
                      verdict=verdict, engine_ok=engine_ok)
        res.rounds.append({"iteration": it, "llm_proposal": p.to_dict(),
                           "sanity": sane,
                           "engine": {"ok": engine_ok,
                                      "raw_error": rec.get("error", "") if not engine_ok else "",
                                      "experiment_id": exp_id,
                                      "metrics": measured,
                                      "result_hash": rec.get("result_hash")},
                           "measured_improvement": imp,
                           "frozen_contract_verdict": verdict,
                           "paired_seed_ci": ci,
                           "llm_finish_decision": dec.to_dict()})
        history.append({"iteration": it, "overrides": p.overrides,
                        "engine_ok": engine_ok, "measured_mse": em,
                        "improvement": imp, "contract_met": verdict["contract_met"],
                        "llm_reasoning": p.reasoning})
        latest = measured or latest
        res.iterations = it + 1
        res.final_improvement = imp
        if dec.finish:
            res.finished = True
            break
        if it + 1 >= max_iterations:
            res.budget_exhausted = True
            break
    res.llm_calls = researcher.n_calls
    (root / "raw_data" / "autonomy" / "result.json").write_text(json.dumps(
        res.to_dict(), indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8")
    (root / "raw_data" / "autonomy" / "search_memory_final.json").write_text(
        json.dumps(memory.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return res


__all__ = ["SCHEMA", "OBJECTIVE", "AutonomousResult", "run_autonomous"]
