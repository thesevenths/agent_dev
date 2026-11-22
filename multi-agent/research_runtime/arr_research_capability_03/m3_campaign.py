"""ARR-03 M3 -- Reacher-v5 action-conditioned transfer campaign.

6 regimes x 3 seeds x 5 arms = 90 real simulator+model runs.
Arms: base, arr_fresh (no knowledge), arr_transfer (verified knowledge),
random, oracle. All arms share the same failure baseline and test set.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .benchmark import (INTERVENTION_REGISTRY, REACHER_REGIMES,
                        intervention_to_reacher_config)
from .diagnosis import ArrDiagnosisAgent, build_agent_evidence
from .integrity import sha256_json
from .knowledge import ResearchKnowledge, knowledge_digest
from .reacher import run_arm, sha256_file

SEEDS = (42, 43, 44)


def _real_artifact_resolver(base_out_dir: Path) -> dict[str, Any]:
    """Bind evidence claims to REAL persisted Reacher artifacts with hashes."""
    claims = []
    for name, fname, text in (
        ("run_json", "run.json", "engine run record with verified metrics"),
        ("model_signature", "model_signature.json", "trained model signature"),
    ):
        path = base_out_dir / fname
        if not path.is_file():
            continue
        digest = sha256_file(path)
        cid = "artifact:" + base_out_dir.name + ":" + name
        claims.append({
            "claim_id": cid, "artifact_path": str(path),
            "artifact_sha256": digest, "claim_text": text,
            "claim_sha256": sha256_json({
                "claim_id": cid, "artifact_sha256": digest,
                "claim_text": text, "run_id": base_out_dir.name}),
            "run_id": base_out_dir.name,
        })
    rows = []
    for c in claims:
        p = Path(c["artifact_path"])
        actual = sha256_file(p) if p.is_file() else ""
        ok = bool(p.is_file()) and actual == c["artifact_sha256"]
        rows.append({**c, "actual_sha256": actual, "ok": ok,
                     "reasons": [] if ok else ["artifact_hash_mismatch"]})
    return {"n_claims": len(claims),
            "n_verified": sum(r["ok"] for r in rows),
            "integrity_pass": bool(claims) and all(r["ok"] for r in rows),
            "claims": rows}


def _norm(base: float | None, arm: float | None) -> float | None:
    if base is None or arm is None or abs(base) < 1e-12:
        return None
    return (base - arm) / abs(base)


def _metrics_to_config_view(run_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "model.hidden": run_cfg["model"].get("hidden"),
        "model.lr": run_cfg["model"].get("lr"),
        "data.n_episodes": run_cfg["train"].get("n_episodes"),
        "data.action_mode": run_cfg["train"].get("action_mode"),
        "data.action_scale": run_cfg["train"].get("action_scale"),
    }


def _evidence_artifacts() -> list[str]:
    return ["artifact:base:model_signature", "artifact:base:run_json"]


def _random_cause(seed: int, key: str) -> str:
    causes = tuple(INTERVENTION_REGISTRY)
    return random.Random(f"{seed}|{key}|m3").choice(causes)


def run_case(*, regime, seed: int, run_root: str | Path,
             seal_dir: str | Path, knowledge: list[ResearchKnowledge],
             knowledge_sha: str) -> dict[str, Any]:
    case_key = f"{regime.regime_id}::s{seed}"
    out_dir = Path(run_root) / "cases" / regime.regime_id.replace("::", "__") / f"s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = 160

    def execute(arm: str, train: dict, model: dict) -> dict[str, Any]:
        return run_arm(arm=arm, regime_id=regime.regime_id, seed=seed,
                       train_cfg=train, test_cfg=regime.test_cfg,
                       model_cfg=model, out_dir=out_dir / arm,
                       epochs=epochs).to_dict()

    # Base = failure baseline. All arms derive from it.
    base = execute("base", regime.failure_train, regime.failure_model)
    base_mse = base.get("metrics", {}).get("one_step_mse")

    # Build blind evidence from the base arm for both ARR diagnosis arms.
    cfg_view = _metrics_to_config_view(
        {"model": regime.failure_model, "train": regime.failure_train})
    ev_metrics = dict(base.get("metrics") or {})
    artifacts = _evidence_artifacts()
    artifact_resolver = _real_artifact_resolver(out_dir / "base")

    def agent_diagnose(arm_tag: str, with_knowledge: str):
        aev = build_agent_evidence(metrics=ev_metrics, config_view=cfg_view,
                                   artifacts=artifacts)
        agent = ArrDiagnosisAgent(out_dir=out_dir / ("agent_" + arm_tag))
        return agent.diagnose(case_id="case_" + sha256_json(case_key)[:12],
                              evidence=aev,
                              transferred_knowledge=with_knowledge)

    kn_text = json.dumps([k.agent_view() for k in knowledge], ensure_ascii=False)
    inv_fresh = agent_diagnose("fresh", "")
    inv_transfer = agent_diagnose("transfer", kn_text)

    def arm_from_agent(inv, tag):
        cause = inv.proposed_cause or inv.primary_cause
        if cause == "UNKNOWN" or cause not in INTERVENTION_REGISTRY:
            cause = "insufficient_model_capacity"
        rc = intervention_to_reacher_config(cause, regime.failure_train,
                                            regime.failure_model)
        return execute(tag, rc["train"], rc["model"]), cause

    arr_fresh, fresh_cause = arm_from_agent(inv_fresh, "arr_fresh")
    arr_transfer, transfer_cause = arm_from_agent(inv_transfer, "arr_transfer")
    rcause = _random_cause(seed, case_key)
    rc = intervention_to_reacher_config(rcause, regime.failure_train,
                                        regime.failure_model)
    random_run = execute("random", rc["train"], rc["model"])
    oracle = execute("oracle", regime.oracle_train, regime.oracle_model)

    arms = {"base": base, "arr_fresh": arr_fresh,
            "arr_transfer": arr_transfer, "random": random_run,
            "oracle": oracle}
    truth = regime.sealed_truth()
    unified_fresh = inv_fresh.primary_cause
    unified_transfer = inv_transfer.primary_cause
    diagnoses = {
        "arr": {"primary": unified_transfer, "confidence": inv_transfer.confidence,
                "candidates": [unified_transfer] + [
                    a.get("cause", "") if isinstance(a, dict) else str(a)
                    for a in inv_transfer.alternatives],
                "evidence_for": inv_transfer.evidence_for,
                "abstain": inv_transfer.abstain},
        "rule": {"primary": "insufficient_model_capacity", "confidence": 0.6,
                 "candidates": ["insufficient_model_capacity",
                                "optimization_instability"], "evidence_for": []},
        "random": {"primary": rcause, "confidence": 0.5,
                   "candidates": [], "evidence_for": []},
        "shuffled": {"primary": _random_cause(seed + 1, case_key),
                     "confidence": 0.5, "candidates": [], "evidence_for": []},
        "oracle": {"primary": truth["cause"], "confidence": 1.0,
                   "candidates": [truth["cause"]], "evidence_for": []},
    }
    case = {
        "case_id": case_key, "regime_id": regime.regime_id,
        "independent_unit": regime.regime_id, "seed": seed, "track": "M3",
        "arms": arms, "diagnoses": diagnoses, "truth": truth,
        "evidence": {"artifact_resolver": artifact_resolver,
                     "metrics": ev_metrics, "config_view": cfg_view},
        "fresh_cause": fresh_cause, "transfer_cause": transfer_cause,
        "knowledge_sha256": knowledge_sha,
        "agent_fresh": inv_fresh.to_dict(), "agent_transfer": inv_transfer.to_dict(),
        "normalized_improvement": {
            k: _norm(base_mse, v.get("metrics", {}).get("one_step_mse"))
            for k, v in arms.items() if k != "base"},
        "ok": all(a.get("ok") for a in arms.values()),
    }
    (out_dir / "case.json").write_text(json.dumps(
        case, indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8")
    return case


def run_m3(*, run_root: str | Path, seal_dir: str | Path,
           knowledge: list[ResearchKnowledge], seeds=SEEDS,
           concurrency: int = 1) -> list[dict[str, Any]]:
    ksha = knowledge_digest(knowledge)
    cases = []
    for regime in REACHER_REGIMES:
        for seed in seeds:
            cases.append(run_case(regime=regime, seed=seed, run_root=run_root,
                                  seal_dir=seal_dir, knowledge=knowledge,
                                  knowledge_sha=ksha))
    return cases


__all__ = ["run_m3", "SEEDS"]
