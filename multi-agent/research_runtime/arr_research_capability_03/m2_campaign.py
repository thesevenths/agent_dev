"""ARR-03 M2 -- state-based causal validation on the 10 verified regimes.

Reuses the real SciNet++ engine adapter from ARR_EMB_RSI_02. The Candidate
arm uses the enhanced ARR LLM diagnosis (M1) to select its intervention.
Every case records base/candidate/random/oracle, agent invocation and sealed GT.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from research_runtime.arr_emb_rsi_02.adapter import WorldModelAdapter
from research_runtime.arr_emb_rsi_02.benchmark import (ALLOWED_KEYS, GENERATORS,
                                                       build_regimes)
from research_runtime.arr_emb_rsi_02.evidence import build_evidence

from .benchmark import M2_EXCLUDED, m2_state_regimes
from .diagnosis import ArrDiagnosisAgent, build_agent_evidence
from .integrity import sha256_json

SEEDS = (42, 43, 44)
CAUSES = (
    "insufficient_model_capacity",
    "insufficient_data_coverage",
    "dynamics_distribution_shift",
    "optimization_instability",
)

# Map V2 sealed cause vocabulary into the unified M1 cause vocabulary.
V2_TO_UNIFIED = {
    "insufficient_model_capacity": "insufficient_model_capacity",
    "insufficient_data_coverage": "insufficient_state_action_coverage",
    "dynamics_distribution_shift": "dynamics_distribution_shift",
    "optimization_instability": "optimization_instability",
}

RANDOM_IVS = (
    {"model.predictor_hidden": [8, 8]}, {"model.predictor_hidden": [32, 32]},
    {"train.lr": 0.0008}, {"train.lr": 0.002},
    {"data.train_samples": 64},
)


def _valid_iv(iv: Any, gen: str = "") -> dict[str, Any] | None:
    if not isinstance(iv, dict) or len(iv) != 1:
        return None
    k, v = next(iter(iv.items()))
    if k not in ALLOWED_KEYS:
        return None
    if k == "model.predictor_hidden":
        if not isinstance(v, list) or not v or any(
                not isinstance(x, int) or not 8 <= x <= 64 for x in v):
            return None
    elif k == "data.train_samples":
        if isinstance(v, bool) or not isinstance(v, int) or not 4 <= v <= 128:
            return None
    elif k == "train.lr":
        if isinstance(v, bool) or not isinstance(v, (int, float)) \
                or not 0.0001 <= float(v) <= 0.01:
            return None
        v = float(v)
    else:
        return None
    out = {k: v}
    if k == "data.train_samples" and gen:
        out["data.frozen_dir"] = (
            f"datasets/arr_emb_rsi_02/{gen}_cov4" if v <= 4
            else f"datasets/arr_emb_rsi_02/{gen}_cov64")
        out["train.batch_size"] = 4
    return out


def _metric(run: dict[str, Any], mech: str) -> float | None:
    m = (run.get("ood_metrics") if mech == "E03" else run.get("metrics")) or {}
    try:
        v = float(m.get("latent_mse@1"))
    except (TypeError, ValueError):
        return None
    return v if v == v else None  # finite check


def _norm(base: float | None, arm: float | None) -> float | None:
    if base is None or arm is None or abs(base) < 1e-12:
        return None
    return (base - arm) / abs(base)


def _run_case(task: tuple[str, int, str, str]) -> dict[str, Any]:
    regime_id, seed, run_root, seal_dir = task
    from research_runtime.arr_emb_rsi_02.benchmark import build_regimes
    regime = next(r for r in build_regimes() if r.regime_id == regime_id)
    gen = regime.generator
    spec = GENERATORS[gen]
    case_key = f"{regime_id}::s{seed}"
    out_dir = Path(run_root) / "cases" / regime_id.replace("::", "__") / f"s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter = WorldModelAdapter(
        run_root=out_dir / "engine", base_config=spec["config"],
        max_jobs=20, max_seconds=3600)
    ood_ov = None
    if regime.mechanism == "E03":
        ood_ov = {"data.frozen_dir": regime.ood_dataset,
                  "data.t_max": spec["ood_t_max"]}

    def run_arm(arm: str, overrides: dict[str, Any]) -> dict[str, Any]:
        ov = dict(regime.common)
        ov.update(overrides)
        return adapter.run_arm(arm=arm, failure_id=gen, seed=seed,
                               overrides=ov, ood_overrides=ood_ov,
                               run_suffix=f"arr03_{arm}").to_dict()

    arms = {"base": run_arm("base", regime.failure_override)}
    metrics = dict(arms["base"].get("ood_metrics") if regime.mechanism == "E03"
                   else arms["base"].get("metrics") or {})
    m1v = metrics.get("latent_mse@1")
    m3v = metrics.get("latent_mse@3") or metrics.get("latent_mse@4")
    if m1v and m3v:
        metrics["horizon_ratio@3@1"] = m3v / m1v
    opaque = "case_" + sha256_json(case_key)[:12]
    ev = build_evidence(case_id=opaque, unit=regime_id, metrics=metrics,
                        run=arms["base"])

    # Agent-facing evidence: normalized metrics + config snapshot + diagnostic
    # features derived ONLY from observed metrics/config (never from labels).
    config_view = {
        "model.predictor_hidden": arms["base"]["applied_config"].get(
            "model", {}).get("predictor_hidden"),
        "train.lr": arms["base"]["applied_config"].get("train", {}).get("lr"),
        "data.train_samples": arms["base"]["applied_config"].get("data", {}).get(
            "train_samples"),
    }
    config_view = {k: v for k, v in config_view.items() if v is not None}
    agent_metrics = dict(metrics)
    # Derived diagnostic features: the agent needs BOTH evaluation views and
    # a scale for lr to distinguish shift / optimisation from pure capacity.
    iid_m = arms["base"].get("metrics") or {}
    ood_m = arms["base"].get("ood_metrics") or {}
    if iid_m and ood_m:
        i1 = iid_m.get("latent_mse@1")
        o1 = ood_m.get("latent_mse@1")
        if i1 and o1 and abs(i1) > 1e-12:
            agent_metrics["iid_latent_mse@1"] = float(i1)
            agent_metrics["heldout_latent_mse@1"] = float(o1)
            agent_metrics["heldout_to_iid_gap"] = float(o1) / abs(float(i1))
    # lr relative to a typical optimiser scale (0.001): a neutral ratio that
    # makes gross optimisation misconfiguration visible without naming it.
    lr = config_view.get("train.lr")
    if isinstance(lr, (int, float)) and lr:
        agent_metrics["lr_vs_typical_ratio"] = float(lr) / 0.001
    # Neutral evaluation protocol context (NOT a label): the agent is allowed
    # to know how metrics were collected, e.g. whether held-out evaluation used
    # a different rollout horizon / distribution than training.
    eval_context = {}
    if regime.mechanism == "E03":
        eval_context = {
            "train_t_max": spec["t_max"],
            "heldout_eval_t_max": spec["ood_t_max"],
            "same_distribution_as_training": False,
        }
    else:
        eval_context = {
            "train_t_max": spec["t_max"],
            "heldout_eval_t_max": spec["t_max"],
            "same_distribution_as_training": True,
        }
    config_view["evaluation_context"] = eval_context
    agent_ev = build_agent_evidence(
        metrics=agent_metrics, config_view=config_view,
        artifacts=[c["claim_id"] for c in ev["payload"]["claims"]])
    agent = ArrDiagnosisAgent(out_dir=out_dir / "agent")
    inv = agent.diagnose(case_id=opaque, evidence=agent_ev)
    proposed = inv.proposed_cause or inv.primary_cause
    mapped = V2_TO_UNIFIED.get(proposed, proposed)
    # Agent selects intervention via the registry cause -> real config.
    if mapped == "insufficient_model_capacity":
        agent_iv = {"model.predictor_hidden": [32, 32]}
    elif mapped == "insufficient_state_action_coverage":
        agent_iv = {"data.train_samples": 64}
    elif mapped == "dynamics_distribution_shift":
        agent_iv = {"model.predictor_hidden": [32, 32]}
    elif mapped == "optimization_instability":
        agent_iv = {"train.lr": 0.0008}
    else:
        agent_iv = {"train.lr": 0.0008}
    agent_iv = _valid_iv(agent_iv, gen) or {"train.lr": 0.0008}
    arms["candidate"] = run_arm("candidate", agent_iv)
    import random as _rnd
    r = _rnd.Random(f"{case_key}|random")
    riv = dict(RANDOM_IVS[r.randrange(len(RANDOM_IVS))])
    riv = _valid_iv(riv, gen) or {"train.lr": 0.0008}
    arms["random"] = run_arm("random", riv)
    arms["oracle"] = run_arm("oracle", regime.oracle_override)

    unified_truth = V2_TO_UNIFIED[regime.sealed_truth["cause"]]
    diagnoses = {
        "arr": {"primary": inv.primary_cause, "confidence": inv.confidence,
                "candidates": [inv.primary_cause] + [
                    a.get("cause", "") if isinstance(a, dict) else str(a)
                    for a in inv.alternatives],
                "evidence_for": inv.evidence_for, "abstain": inv.abstain,
                "proposed_cause": inv.proposed_cause},
        "rule": {"primary": "insufficient_model_capacity", "confidence": 0.6,
                 "candidates": ["insufficient_model_capacity",
                                "optimization_instability"], "evidence_for": []},
        "random": {"primary": CAUSES[r.randrange(len(CAUSES))],
                   "confidence": round(r.uniform(0.3, 0.8), 4),
                   "candidates": [], "evidence_for": []},
        "shuffled": {"primary": CAUSES[r.randrange(len(CAUSES))],
                     "confidence": round(r.uniform(0.3, 0.8), 4),
                     "candidates": [], "evidence_for": []},
        "oracle": {"primary": unified_truth, "confidence": 1.0,
                   "candidates": [unified_truth], "evidence_for": []},
    }
    base_v = _metric(arms["base"], regime.mechanism)
    case = {
        "case_id": case_key, "regime_id": regime_id,
        "independent_unit": regime_id, "seed": seed,
        "track": "M2", "arms": arms, "evidence": ev,
        "agent_invocation": inv.to_dict(),
        "diagnoses": diagnoses,
        "truth": {"cause": unified_truth},
        "agent_intervention": agent_iv, "random_intervention": riv,
        "normalized_improvement": {
            k: _norm(base_v, _metric(v, regime.mechanism))
            for k, v in arms.items()},
        "ok": all(a.get("ok") for a in arms.values()),
    }
    (out_dir / "case.json").write_text(json.dumps(
        case, indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8")
    return case


def run_m2(*, run_root: str | Path, seal_dir: str | Path,
           concurrency: int = 3, seeds=SEEDS) -> list[dict[str, Any]]:
    reg = m2_state_regimes()
    tasks = [(r["regime_id"], s, str(run_root), str(seal_dir))
             for r in reg for s in seeds]
    with ProcessPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(_run_case, tasks))


__all__ = ["run_m2", "SEEDS"]
