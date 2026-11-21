"""ARR-RSI-01 P0.1 -- diagnosis inferred ONLY from observed evidence.

No task family, preset failure mode or sealed intervention value is read here.
When evidence is insufficient the diagnoser returns an explicit
``insufficient_evidence`` result and proposes NO intervention.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..failure_diagnoser import FailureDiagnoser
from ..schemas import ExperimentResult
from .tasks import ALLOWED_CONFIG_SCHEMA, assert_no_ground_truth, sha256_json

REQUIRED_DIAGNOSIS_FIELDS = (
    "failure_modes", "root_cause", "evidence_refs", "confidence",
    "alternative_explanations", "improvement_hypothesis",
    "proposed_intervention", "expected_metric", "falsification_condition",
    "diagnosis_id", "insufficient_evidence", "evidence_basis",
)
INSUFFICIENT = "insufficient_evidence"
CAUSE_HORIZON = "dynamics_error_accumulation"
CAUSE_CAPACITY = "representation_capacity_limited"
CAUSE_OOD = "ood_generalization_gap"
CAUSE_WEAK = "weak_prediction_signal"
CAUSE_INFRA = "infrastructure_failure"

CAUSE_PLAN: dict[str, dict[str, Any]] = {
    CAUSE_HORIZON: {"key": "train.lr", "direction": "decrease",
                    "mechanism": "error accumulates over rollout; a smaller "
                                 "optimisation step may stabilise latent dynamics"},
    CAUSE_CAPACITY: {"key": "model.predictor_hidden", "direction": "increase",
                     "mechanism": "poor latent similarity suggests limited "
                                  "predictor capacity"},
    CAUSE_OOD: {"key": "model.latent_dim", "direction": "increase",
                "mechanism": "validation/OOD gap suggests latent bottleneck "
                             "under regime shift"},
    CAUSE_WEAK: {"key": "jepa.pred_weight", "direction": "increase",
                 "mechanism": "flat but weak horizons suggest an underweighted "
                              "prediction objective"},
}
INFRA_ERROR_TOKENS = (
    "worker_interrupted", "AdapterError", "TimeoutError", "timed out",
    "KeyError", "engine CLI error", "incomplete", "already exist",
    "ConfigurationError", "RuntimeError",
)


@dataclass(frozen=True)
class DiagnosisOutput:
    task_id: str
    failure_modes: list[str]
    root_cause: str
    evidence_refs: list[str]
    confidence: float
    alternative_explanations: list[str]
    improvement_hypothesis: str
    proposed_intervention: dict[str, Any] | None
    expected_metric: dict[str, Any]
    falsification_condition: str
    diagnosis_id: str = ""
    insufficient_evidence: bool = False
    evidence_basis: dict[str, Any] = field(default_factory=dict)
    shuffled_from: str = ""
    diagnosis_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if not d.get("diagnosis_sha256"):
            d["diagnosis_sha256"] = sha256_json(d)
        return d

    def validate(self) -> list[str]:
        d = self.to_dict()
        return [k for k in REQUIRED_DIAGNOSIS_FIELDS
                if k not in d or d[k] in (None, "", [], {})]


@runtime_checkable
class Diagnosis(Protocol):
    def diagnose(self, *, task: dict[str, Any], trajectory: dict[str, Any],
                 experiment_results: list[dict[str, Any]],
                 evidence: dict[str, Any],
                 budget: dict[str, Any]) -> DiagnosisOutput: ...


def _num(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _dotted(config: dict[str, Any], key: str, default: Any = None) -> Any:
    node: Any = config
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _perturb(*, key: str, baseline_config: dict[str, Any], direction: str) -> Any:
    """Generic type-aware perturbation; NO benchmark value is consulted."""
    base = _dotted(baseline_config, key, None)
    schema = next((s for s in ALLOWED_CONFIG_SCHEMA if s["key"] == key), None)
    if base is None or isinstance(base, bool):
        if not schema:
            return None
        base = 1.0
    factor = 0.8 if direction == "decrease" else 1.25
    if isinstance(base, list) and base and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in base):
        return [max(1, int(round(v * factor))) if isinstance(v, int)
                else float(v * factor) for v in base]
    if isinstance(base, int):
        return max(1, int(round(base * factor)))
    if isinstance(base, float):
        return round(base * factor, 8)
    return None


def _metric_index(results: list[dict[str, Any]],
                  trajectory: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    sources = list(results) + list(trajectory.get("experiment_results") or [])
    for item in sources:
        if not isinstance(item, dict):
            continue
        for key, value in (item.get("metrics") or {}).items():
            v = _num(value)
            if v is not None:
                out.setdefault(str(key), v)
    return out


def _horizon_ratio(metrics: dict[str, float]) -> float | None:
    groups: dict[str, list[tuple[int, float]]] = {}
    for key, value in metrics.items():
        m = re.match(r"^(.+)@(\d+)$", key)
        if m:
            groups.setdefault(m.group(1), []).append((int(m.group(2)), value))
    for name, points in groups.items():
        if len(points) < 2 or not any(
                t in name.lower() for t in ("mse", "loss", "error", "rmse")):
            continue
        points.sort()
        if points[0][1] > 0:
            return points[-1][1] / points[0][1]
    return None


def _evidence_refs(*, trajectory: dict[str, Any],
                   experiment_results: list[dict[str, Any]]) -> list[str]:
    refs: set[str] = set()

    def add(prefix: str, value: Any) -> None:
        if isinstance(value, str) and value:
            refs.add(f"{prefix}:{value}")

    for key in ("baseline_id", "report_path", "result_path", "log_path",
                "applied_config_path"):
        add(key, trajectory.get(key))
    evidence = trajectory.get("evidence_refs") or {}
    if isinstance(evidence, dict):
        for key, value in evidence.items():
            add(key, value if isinstance(value, str) else sha256_json(value)[:16])
    for item in experiment_results:
        if not isinstance(item, dict):
            continue
        for key in ("experiment_id", "job_id", "result_path", "log_path"):
            add(key, item.get(key))
    for call in (trajectory.get("tool_calls") or []):
        if isinstance(call, dict):
            add("tool_call", call.get("call_id"))
    return sorted(refs)


class EvidenceGroundedDiagnoser:
    """Deterministic cause inference from measured evidence only."""

    def __init__(self, diagnoser: FailureDiagnoser | None = None):
        self._rules = diagnoser or FailureDiagnoser()

    def diagnose(self, *, task: dict[str, Any], trajectory: dict[str, Any],
                 experiment_results: list[dict[str, Any]],
                 evidence: dict[str, Any],
                 budget: dict[str, Any]) -> DiagnosisOutput:
        assert_no_ground_truth(task)
        refs = _evidence_refs(trajectory=trajectory,
                              experiment_results=experiment_results)
        metrics = _metric_index(experiment_results, trajectory)
        errors = " ".join(str(item.get("error") or "") for item in
                          experiment_results if isinstance(item, dict))
        matched_rules: list[str] = []
        for raw in experiment_results:
            if not isinstance(raw, dict):
                continue
            payload = {k: v for k, v in raw.items()
                       if k in ExperimentResult.model_fields}
            payload.setdefault("experiment_id", "unknown")
            payload.setdefault("job_id", "unknown")
            payload.setdefault("status", raw.get("status") or "incomplete")
            try:
                for report in self._rules.diagnose_all(ExperimentResult(**payload)):
                    matched_rules.append(report.category)
            except Exception:
                continue

        baseline_config = dict(trajectory.get("baseline_config") or {}) \
            if isinstance(trajectory.get("baseline_config"), dict) else {}
        ratio = _horizon_ratio(metrics)
        cosine = [_num(v) for k, v in metrics.items() if "cos" in k.lower()]
        cosine = [v for v in cosine if v is not None]
        mean_cos = sum(cosine) / len(cosine) if cosine else None
        final = trajectory.get("final_conclusion") or {}
        holdout = _num((final.get("holdout") or {}).get("mean_holdout_delta"))
        ood = _num((final.get("ood") or {}).get("mean_holdout_delta"))
        ood_gap = holdout - ood if (holdout is not None and ood is not None) else None
        basis = {"n_metric_keys": len(metrics), "horizon_ratio": ratio,
                 "mean_cosine": mean_cos, "ood_gap": ood_gap,
                 "n_evidence_refs": len(refs),
                 "matched_rules": sorted(set(matched_rules)),
                 "n_results": len(experiment_results),
                 "budget_used": (trajectory.get("budget_usage") or {}).get("jobs_used")}

        cause = ""
        if not refs or not metrics:
            cause = INSUFFICIENT
        elif any(token in errors for token in INFRA_ERROR_TOKENS) or \
                any(r in ("worker_lost", "process_failure", "budget_exceeded",
                          "result_unavailable", "pipeline_incomplete")
                    for r in matched_rules):
            cause = CAUSE_INFRA
        elif ood_gap is not None and ood_gap > 0.05:
            cause = CAUSE_OOD
        elif ratio is not None and ratio > 1.08:
            cause = CAUSE_HORIZON
        elif mean_cos is not None and mean_cos < 0.6:
            cause = CAUSE_CAPACITY
        elif metrics:
            cause = CAUSE_WEAK

        insufficient = cause in (INSUFFICIENT, CAUSE_INFRA)
        plan = CAUSE_PLAN.get(cause, {})
        key = str(plan.get("key") or "")
        value = _perturb(key=key, baseline_config=baseline_config,
                         direction=str(plan.get("direction") or "increase")) \
            if key else None
        intervention = None
        if not insufficient and key and value is not None:
            intervention = {"key": key, "value": value,
                            "direction": plan.get("direction"),
                            "mechanism": plan.get("mechanism")}

        confidence = 0.0
        if not insufficient:
            confidence = 0.55
            if len(refs) >= 3 and len(metrics) >= 2:
                confidence = 0.75
            if ratio is not None or ood_gap is not None or mean_cos is not None:
                confidence = 0.82
        payload = {
            "task_id": str(task.get("agent_task_id") or task.get("task_id") or ""),
            "failure_modes": sorted(set(matched_rules)) or [cause],
            "root_cause": cause, "evidence_refs": refs[:50],
            "confidence": round(confidence, 4),
            "alternative_explanations": [
                "measurement noise or an engine-side artifact rather than a model effect",
                "the frozen split is not representative of the observed metric pattern",
                "budget truncation produced an unfalsifiable result",
            ],
            "improvement_hypothesis": (
                f"Observed evidence {basis} supports {cause}. "
                + (f"A {plan.get('direction')} change to {key} may improve "
                   f"{task.get('primary_metric', 'the primary metric')}."
                   if intervention else "No intervention is justified by this evidence.")),
            "proposed_intervention": intervention,
            "expected_metric": {
                "name": task.get("primary_metric", ""),
                "direction": task.get("metric_direction", "minimize"),
                "minimum_meaningful_delta": 0.02, "evaluation_split": "validation",
            },
            "falsification_condition": (
                f"Reject if the matched-budget arm changing only {key} does not "
                f"improve the reserved split, or if any infrastructure failure "
                f"occurs in that arm." if key else
                "No candidate is proposed; result remains INSUFFICIENT_EVIDENCE."),
            "insufficient_evidence": insufficient,
            "evidence_basis": basis, "shuffled_from": "", "diagnosis_sha256": "",
        }
        digest = sha256_json(payload)
        out = DiagnosisOutput(diagnosis_id=f"dg_{digest[:16]}", **payload)
        assert_no_ground_truth(out.to_dict())
        return out


class ShuffledDiagnoser:
    """Attribution control using ANOTHER TASK'S REAL diagnosis."""

    def __init__(self, *, pool: dict[str, DiagnosisOutput], seed: int = 20261121):
        self._pool = dict(pool)
        self._seed = int(seed)

    def diagnose(self, *, task: dict[str, Any], trajectory: dict[str, Any],
                 experiment_results: list[dict[str, Any]],
                 evidence: dict[str, Any],
                 budget: dict[str, Any]) -> DiagnosisOutput:
        own_id = str(task.get("agent_task_id") or task.get("task_id") or "")
        candidates = [k for k in sorted(self._pool) if k != own_id]
        refs = _evidence_refs(trajectory=trajectory,
                              experiment_results=experiment_results)
        index = int.from_bytes(hashlib.sha256(
            f"{self._seed}|{own_id}".encode()).digest()[:4], "big")
        if not candidates:
            alternatives = [c for c in CAUSE_PLAN]
            cause = alternatives[index % len(alternatives)]
            plan = CAUSE_PLAN[cause]
            marker = "synthetic_shuffle"
        else:
            src = self._pool[candidates[index % len(candidates)]]
            cause = src.root_cause
            plan = CAUSE_PLAN.get(cause, {})
            marker = f"shuffled_from:{src.diagnosis_id}"
        baseline_config = dict(trajectory.get("baseline_config") or {}) \
            if isinstance(trajectory.get("baseline_config"), dict) else {}
        key = str(plan.get("key") or "")
        value = _perturb(key=key, baseline_config=baseline_config,
                         direction=str(plan.get("direction") or "increase")) \
            if key else None
        payload = {
            "task_id": own_id, "failure_modes": [cause], "root_cause": cause,
            "evidence_refs": sorted(set(refs + [marker]))[:50], "confidence": 0.0,
            "alternative_explanations": ["CONTROL ONLY: diagnosis not derived "
                                         "from this task's evidence"],
            "improvement_hypothesis": "CONTROL ONLY: shuffled diagnosis; no claim.",
            "proposed_intervention": ({"key": key, "value": value,
                                       "direction": plan.get("direction"),
                                       "mechanism": "shuffled control"}
                                      if value is not None else None),
            "expected_metric": {}, "falsification_condition": "CONTROL ONLY.",
            "insufficient_evidence": False,
            "evidence_basis": {"shuffle": marker}, "shuffled_from": marker,
            "diagnosis_sha256": "",
        }
        digest = sha256_json(payload)
        out = DiagnosisOutput(diagnosis_id=f"dg_shuffle_{digest[:12]}", **payload)
        assert_no_ground_truth(out.to_dict())
        return out


__all__ = [
    "Diagnosis", "DiagnosisOutput", "EvidenceGroundedDiagnoser",
    "ShuffledDiagnoser", "REQUIRED_DIAGNOSIS_FIELDS", "INSUFFICIENT",
    "CAUSE_HORIZON", "CAUSE_CAPACITY", "CAUSE_OOD", "CAUSE_WEAK", "CAUSE_INFRA",
]
