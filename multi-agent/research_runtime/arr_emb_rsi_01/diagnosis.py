"""ARR_EMB_RSI_01 -- diagnosis benchmark controllers and intervention registry."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace as dc_replace
from typing import Any

from .benchmark import FAILURES
from .integrity import assert_no_ground_truth, sha256_json

CONTROLLERS = ("evidence", "random", "shuffled", "rule", "oracle")
ROOT_CAUSES = tuple(sorted({f["root_cause"] for f in FAILURES.values()}))


@dataclass(frozen=True)
class DiagnosisOutput:
    case_id: str
    controller: str
    primary_hypothesis: str
    root_cause: str
    top2: tuple[str, ...]
    confidence: float
    evidence: tuple[str, ...]
    alternatives: tuple[str, ...]
    proposed_intervention: dict[str, Any] | None
    expected_effect: str
    falsification_condition: str
    budget: dict[str, Any]
    diagnosis_id: str = ""
    abstain: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["top2"] = list(self.top2)
        d["evidence"] = list(self.evidence)
        d["alternatives"] = list(self.alternatives)
        if not d["diagnosis_id"]:
            d["diagnosis_id"] = "dg_" + sha256_json(d)[:16]
        return d


def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def _evidence_cause(evidence: dict[str, Any]) -> tuple[str, tuple[str, ...], float]:
    """Infer from metrics/config ONLY. No failure_id, no GT field."""
    metrics = evidence.get("metrics") or {}
    snap = evidence.get("config_snapshot") or {}
    ratio = _num(metrics.get("horizon_ratio@3@1"))
    cos1 = _num(metrics.get("cosine@1"))
    ood = any(str(k).startswith("ood_") for k in metrics)
    keys = [str(k) for k in snap]
    if ood:
        return "dynamics_distribution_shift", (
            "dynamics_distribution_shift", "insufficient_model_capacity"), 0.78
    if any("train_samples" in k for k in keys):
        return "insufficient_state_action_coverage", (
            "insufficient_state_action_coverage", "insufficient_model_capacity"), 0.76
    if ratio is not None and ratio > 1.05:
        return "long_horizon_error_accumulation", (
            "long_horizon_error_accumulation", "weak_prediction_objective"), 0.74
    if cos1 is not None and cos1 < 0.3:
        return "insufficient_model_capacity", (
            "insufficient_model_capacity", "optimisation_budget_too_small"), 0.72
    return "", (), 0.0


def _intervention(cause: str) -> dict[str, Any] | None:
    reg = INTERVENTION_REGISTRY.get(cause)
    return {"intervention_id": reg["intervention_id"],
            "overrides": dict(reg["candidate_overrides"])} if reg else None


def diagnose(*, controller: str, evidence: dict[str, Any],
             sealed: dict[str, Any] | None = None,
             shuffle_pool: list[dict[str, Any]] | None = None,
             seed: int = 20261122) -> DiagnosisOutput:
    assert_no_ground_truth(evidence)
    if controller not in CONTROLLERS:
        raise ValueError(f"unknown controller {controller}")
    case_id = str(evidence.get("case_id") or "unknown")
    refs = tuple(sorted(str(c.get("artifact_path") or c.get("claim_id"))
                        for c in evidence.get("claims") or []))
    if controller == "oracle":
        if not sealed:
            raise ValueError("Oracle requires sealed ground truth")
        cause = str(sealed["ground_truth"])
        top2 = (cause, str((sealed.get("alternatives") or [""])[0]))
        conf = 1.0
        iv = {"intervention_id": f"oracle_{cause}",
              "overrides": dict(sealed.get("correct_intervention") or {})}
        alts = tuple(sealed.get("alternatives") or ())
        expected = str(sealed.get("expected_metric") or "")
        falsification = str(sealed.get("falsification_condition") or "")
    elif controller == "random":
        digest = int.from_bytes(hashlib.sha256(
            f"{seed}|{case_id}".encode()).digest()[:4], "big")
        cause = ROOT_CAUSES[digest % len(ROOT_CAUSES)]
        top2 = (cause, ROOT_CAUSES[(digest + 1) % len(ROOT_CAUSES)])
        conf = 0.5
        iv = _intervention(cause)
        alts = tuple(c for c in ROOT_CAUSES if c != cause)[:2]
        expected, falsification = "uncontrolled random prediction", "n/a"
    elif controller == "shuffled":
        if not shuffle_pool:
            raise ValueError("Shuffled controller requires a pool")
        idx = int.from_bytes(hashlib.sha256(
            f"{seed}|{case_id}".encode()).digest()[:4], "big") % len(shuffle_pool)
        donor = shuffle_pool[idx]
        cause = str(donor.get("root_cause") or "")
        top2 = tuple(donor.get("top2") or (cause,))
        conf = float(donor.get("confidence") or 0.5)
        iv = donor.get("proposed_intervention")
        alts = tuple(donor.get("alternatives") or ())
        expected = "evidence correspondence destroyed by shuffle"
        falsification = "control only"
        refs = tuple(list(refs) + [f"shuffled_from:{donor.get('diagnosis_id','')}"])
    else:
        cause, top2, conf = _evidence_cause(evidence)
        iv = _intervention(cause) if cause else None
        alts = tuple(c for c in ROOT_CAUSES if c != cause)[:2]
        expected = "reduce primary prediction error under matched budget"
        falsification = ("No significant improvement on validation/OOD metrics "
                         "after the single registered intervention.")
        if controller == "rule" and refs:
            refs = ()  # rule-based baseline uses no artifact-level evidence refs
    abstain = not bool(cause) and controller in ("evidence", "rule")
    if abstain:
        iv = None
    out = DiagnosisOutput(
        case_id=case_id, controller=controller,
        primary_hypothesis=(f"Evidence supports {cause}." if cause else
                            "Evidence insufficient; abstain."),
        root_cause=cause, top2=top2, confidence=round(conf, 4),
        evidence=refs, alternatives=alts, proposed_intervention=iv,
        expected_effect=expected, falsification_condition=falsification,
        budget={"max_jobs": 4}, abstain=abstain)
    payload = out.to_dict()
    assert_no_ground_truth(payload)
    return dc_replace(out, diagnosis_id=payload["diagnosis_id"])


INTERVENTION_REGISTRY: dict[str, dict[str, Any]] = {
    "insufficient_model_capacity": {
        "intervention_id": "IV-CAP-01",
        "target_root_cause": "insufficient_model_capacity",
        "engine_parameters": {"model.predictor_hidden": "predictor MLP widths"},
        "baseline_value": [8, 8], "candidate_value": [16, 16],
        "candidate_overrides": {"model.predictor_hidden": [16, 16]},
        "expected_mechanism": "larger predictor can represent multi-step latent dynamics",
        "expected_metric": "latent_mse@1", "expected_direction": "minimize",
        "minimum_effect": 0.02, "cost": {"jobs": 2},
        "falsification": "No validation improvement with hidden=[16,16] vs [8,8].",
        "source": "engine_verified_sweep_20261122"},
    "insufficient_state_action_coverage": {
        "intervention_id": "IV-COV-01",
        "target_root_cause": "insufficient_state_action_coverage",
        "engine_parameters": {"data.train_samples": "training split size"},
        "baseline_value": 16, "candidate_value": 64,
        "candidate_overrides": {"data.train_samples": 64,
                                "data.frozen_dir": "datasets/arr_emb_rsi_coverage_64"},
        "expected_mechanism": "more state coverage reduces predictable dynamics error",
        "expected_metric": "latent_mse@1", "expected_direction": "minimize",
        "minimum_effect": 0.02, "cost": {"jobs": 2},
        "falsification": "No improvement with 64 samples under fixed hidden test.",
        "source": "engine_verified_setup_20261122"},
    "dynamics_distribution_shift": {
        "intervention_id": "IV-SHIFT-01",
        "target_root_cause": "dynamics_distribution_shift",
        "engine_parameters": {"model.predictor_hidden": "predictor MLP widths"},
        "baseline_value": [8, 8], "candidate_value": [16, 16],
        "candidate_overrides": {"model.predictor_hidden": [16, 16]},
        "expected_mechanism": "greater capacity improves transfer to shifted t_max regime",
        "expected_metric": "ood_latent_mse@1", "expected_direction": "minimize",
        "minimum_effect": 0.02, "cost": {"jobs": 3},
        "falsification": "No OOD improvement with hidden=[16,16].",
        "source": "engine_verified_setup_20261122"},
    "long_horizon_error_accumulation": {
        "intervention_id": "IV-HOR-01",
        "target_root_cause": "long_horizon_error_accumulation",
        "engine_parameters": {"jepa.rollout_steps": "multi-step prediction horizons"},
        "baseline_value": [1], "candidate_value": [1, 2, 3],
        "candidate_overrides": {"jepa.rollout_steps": [1, 2, 3]},
        "expected_mechanism": "multi-step loss suppresses horizon error accumulation",
        "expected_metric": "horizon_ratio@3@1", "expected_direction": "minimize",
        "minimum_effect": 0.05, "cost": {"jobs": 2},
        "falsification": "No reduction in horizon ratio with rollout_steps=[1,2,3].",
        "source": "engine_verified_setup_20261122"},
}


__all__ = ["CONTROLLERS", "ROOT_CAUSES", "DiagnosisOutput", "diagnose",
           "INTERVENTION_REGISTRY"]
