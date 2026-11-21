"""ARR-RSI-01 -- minimal, evidence-grounded Diagnosis interface.

The interface deliberately does NOT accept ground truth. ``EvidenceGroundedDiagnoser``
reads only the public task view, the recorded trajectory, structured experiment
results and the evidence index. ``ShuffledDiagnoser`` is an attribution control:
it keeps the real evidence but swaps the diagnosis with another task's, so any
apparent improvement attributable to "diagnosis quality" disappears.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..failure_diagnoser import FailureDiagnoser
from ..schemas import ExperimentResult
from .tasks import assert_no_ground_truth, sha256_json

REQUIRED_DIAGNOSIS_FIELDS = (
    "failure_modes", "root_cause", "evidence_refs", "confidence",
    "alternative_explanations", "improvement_hypothesis",
    "proposed_intervention", "expected_metric", "falsification_condition",
)

# Generic, framework-independent cause vocabulary. Nothing here names a
# benchmark target, a domain-specific class or a ground-truth field.
CAUSE_CAPACITY = "insufficient_representation_capacity"
CAUSE_DYNAMICS = "latent_dynamics_undertrained"
CAUSE_OBJECTIVE = "weak_prediction_objective"
CAUSE_IDENTIFIABILITY = "parameter_identifiability_loss"
CAUSE_EVIDENCE_GAP = "evidence_gap"
CAUSE_PIPELINE = "measurement_pipeline_defect"

_FAILURE_TO_CAUSE = {
    "dynamics_error_accumulation": CAUSE_DYNAMICS,
    "long_horizon_state_drift": CAUSE_DYNAMICS,
    "parameter_identifiability_loss": CAUSE_IDENTIFIABILITY,
    "conservation_signal_weak": CAUSE_OBJECTIVE,
    "ood_regime_shift": CAUSE_CAPACITY,
}

# Evidence-derived causes override the failure-mode prior when present.
_EVIDENCE_CAUSE = {
    "horizon_degradation": CAUSE_DYNAMICS,
    "metric_quality": CAUSE_PIPELINE,
    "no_metrics": CAUSE_EVIDENCE_GAP,
    "result_unavailable": CAUSE_EVIDENCE_GAP,
    "pipeline_incomplete": CAUSE_PIPELINE,
    "process_failure": CAUSE_PIPELINE,
    "budget_exceeded": CAUSE_EVIDENCE_GAP,
    "worker_lost": CAUSE_PIPELINE,
}


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
    diagnosis_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if not d.get("diagnosis_sha256"):
            d["diagnosis_sha256"] = sha256_json(d)
        return d

    def validate(self) -> list[str]:
        d = self.to_dict()
        missing = [k for k in REQUIRED_DIAGNOSIS_FIELDS
                   if k not in d or d[k] in (None, "", [], {})]
        if not 0.0 <= float(self.confidence) <= 1.0:
            missing.append("confidence_range")
        return missing


@runtime_checkable
class Diagnosis(Protocol):
    """Diagnosis agent interface. No ground-truth parameter, by design."""

    def diagnose(self, *, task: dict[str, Any], trajectory: dict[str, Any],
                 experiment_results: list[dict[str, Any]],
                 evidence: dict[str, Any],
                 budget: dict[str, Any]) -> DiagnosisOutput: ...


def _as_experiment_result(raw: dict[str, Any]) -> ExperimentResult:
    """Project a recorded engine result into the existing structured schema."""
    allowed = set(ExperimentResult.model_fields)
    payload = {k: v for k, v in raw.items() if k in allowed}
    payload.setdefault("experiment_id", str(raw.get("experiment_id", "unknown")))
    payload.setdefault("job_id", str(raw.get("job_id", "unknown")))
    payload.setdefault("status", raw.get("status") or "incomplete")
    return ExperimentResult(**payload)


def _collect_evidence_refs(*, trajectory: dict[str, Any],
                           experiment_results: list[dict[str, Any]],
                           evidence: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for key in ("experiment_id", "baseline_id", "report_path", "result_path"):
        value = trajectory.get(key) or evidence.get(key)
        if value:
            refs.append(f"{key}:{value}")
    for item in experiment_results:
        for key in ("experiment_id", "job_id", "result_path", "log_path"):
            value = item.get(key)
            if value:
                refs.append(f"{key}:{value}")
    for item in (evidence.get("artifacts") or []):
        if isinstance(item, dict):
            for key in ("path", "sha256", "artifact_id"):
                if item.get(key):
                    refs.append(f"artifact:{item[key]}")
        elif item:
            refs.append(f"artifact:{item}")
    for call in (trajectory.get("tool_calls") or []):
        if isinstance(call, dict) and call.get("call_id"):
            refs.append(f"tool_call:{call['call_id']}")
    # Stable, de-duplicated ordering. Refs point only to RECORDED artifacts.
    return sorted(set(str(r) for r in refs))


class EvidenceGroundedDiagnoser:
    """Deterministic diagnosis from recorded evidence only."""

    def __init__(self, *, diagnoser: FailureDiagnoser | None = None):
        self._diagnoser = diagnoser or FailureDiagnoser()

    def diagnose(self, *, task: dict[str, Any], trajectory: dict[str, Any],
                 experiment_results: list[dict[str, Any]],
                 evidence: dict[str, Any],
                 budget: dict[str, Any]) -> DiagnosisOutput:
        # Fail closed if an upstream caller accidentally supplies GT fields.
        assert_no_ground_truth(task)
        task_id = str(task.get("task_id", "unknown"))
        failure_mode = str(task.get("failure_mode", ""))
        refs = _collect_evidence_refs(trajectory=trajectory,
                                      experiment_results=experiment_results,
                                      evidence=evidence)
        matched: list[str] = []
        reports: list[dict[str, Any]] = []
        for raw in experiment_results:
            try:
                result = _as_experiment_result(raw)
            except Exception:
                continue
            for report in self._diagnoser.diagnose_all(result):
                matched.append(str(report.category))
                reports.append({"category": report.category,
                                "layer": report.layer,
                                "hypothesis": report.hypothesis or "",
                                "evidence": report.evidence})
                for key in ("experiment_id", "job_id"):
                    value = getattr(report, key, "")
                    if value:
                        refs.append(f"failure_report:{key}:{value}")

        # Cause selection: evidence first, failure-mode prior second. An
        # unmatched failure mode is an explicit evidence gap, never a guess.
        cause = ""
        for rule_name, mapped in _EVIDENCE_CAUSE.items():
            if any(rule_name in m or m in str(reports) for m in matched):
                cause = mapped
                break
        if not cause:
            cause = _FAILURE_TO_CAUSE.get(failure_mode, CAUSE_EVIDENCE_GAP)
        if not refs:
            cause = CAUSE_EVIDENCE_GAP

        confidence = 0.35
        if refs:
            confidence = 0.55
        if matched and refs:
            confidence = 0.82
        if len(set(matched)) >= 2 and len(refs) >= 3:
            confidence = 0.9

        candidates = [iv for iv in (task.get("valid_interventions") or [])
                      if isinstance(iv, dict) and iv.get("role") == "candidate"]
        intervention = dict(candidates[0]) if candidates else None
        if cause == CAUSE_EVIDENCE_GAP or not refs:
            intervention = None
        if cause == CAUSE_PIPELINE:
            intervention = None

        metric = str(task.get("primary_metric") or "latent_mse@1")
        hypothesis = (
            f"Evidence ({len(refs)} refs, {len(matched)} matched failure rules) "
            f"indicates {cause}; a single controlled change to "
            f"{intervention.get('key') if intervention else 'the measurement pipeline'} "
            f"should improve {metric} on the matched validation split."
        )
        alternatives = [
            "measurement noise or an engine-side artifact rather than a model effect",
            "the frozen split is not representative of the stated failure mode",
            "budget truncation produced an unfalsifiable result",
        ]
        if intervention is None:
            alternatives.insert(
                0, "insufficient evidence to attribute the failure to a method knob")
        falsification = (
            f"Reject if a matched-budget arm changing only "
            f"{intervention.get('key') if intervention else 'an inert control'} "
            f"does not improve {metric} on the reserved split, or if any "
            f"infrastructure failure occurred in the arm."
        )
        out = DiagnosisOutput(
            task_id=task_id,
            failure_modes=sorted(set(matched)) or ([failure_mode] if failure_mode else ["unclassified"]),
            root_cause=cause,
            evidence_refs=refs[:50],
            confidence=round(float(confidence), 4),
            alternative_explanations=alternatives,
            improvement_hypothesis=hypothesis,
            proposed_intervention=intervention,
            expected_metric={"name": metric,
                             "direction": str(task.get("metric_direction") or "minimize"),
                             "minimum_meaningful_delta": 0.02,
                             "evaluation_split": "validation"},
            falsification_condition=falsification,
        )
        out_dict = out.to_dict()
        assert_no_ground_truth(out_dict)
        return out


class ShuffledDiagnoser:
    """Attribution control: real evidence, permuted diagnosis.

    ``pool`` is a deterministic mapping task_id -> DiagnosisOutput produced by
    the evidence-grounded diagnoser on OTHER tasks. The shuffled diagnosis is
    never used to make a scientific claim; it exists only to test whether the
    Candidate's gain depends on the diagnosis being correct.
    """

    def __init__(self, *, pool: dict[str, DiagnosisOutput], seed: int = 20261121):
        self._pool = dict(pool)
        self._seed = int(seed)

    def diagnose(self, *, task: dict[str, Any], trajectory: dict[str, Any],
                 experiment_results: list[dict[str, Any]],
                 evidence: dict[str, Any],
                 budget: dict[str, Any]) -> DiagnosisOutput:
        task_id = str(task.get("task_id", "unknown"))
        others = [k for k in sorted(self._pool) if k != task_id]
        if not others:
            raise ValueError("shuffled diagnosis requires >=2 diagnosed tasks")
        index = (int.from_bytes(
            __import__("hashlib").sha256(
                f"{self._seed}|{task_id}".encode("utf-8")).digest()[:4],
            "big")) % len(others)
        source = self._pool[others[index]]
        d = source.to_dict()
        d["task_id"] = task_id
        d["evidence_refs"] = sorted(set(
            [f"shuffled_from:{source.task_id}"] +
            _collect_evidence_refs(trajectory=trajectory,
                                   experiment_results=experiment_results,
                                   evidence=evidence)))[:50]
        d["improvement_hypothesis"] = (
            "CONTROL ONLY: diagnosis shuffled from another task; no claim made.")
        d["diagnosis_sha256"] = ""
        assert_no_ground_truth(d)
        return DiagnosisOutput(**{k: d[k] for k in DiagnosisOutput.__dataclass_fields__
                                  if k in d})


__all__ = [
    "Diagnosis", "DiagnosisOutput", "EvidenceGroundedDiagnoser",
    "ShuffledDiagnoser", "REQUIRED_DIAGNOSIS_FIELDS",
    "CAUSE_CAPACITY", "CAUSE_DYNAMICS", "CAUSE_OBJECTIVE",
    "CAUSE_IDENTIFIABILITY", "CAUSE_EVIDENCE_GAP", "CAUSE_PIPELINE",
]
