"""
research_runtime.schemas
========================
The single data contract between the Agent Runtime and ANY scientific engine.

Execution contract (P0):

    ResearchObjective   what are we actually trying to prove?
    ExperimentSpec      what exactly was run?
    ExperimentResult    what came out (success OR failure)?
    FailureReport       why did it fail / what should be tested next?
    ModelVersion        which checkpoint, which data, which parent, promoted?

Research decision contract (P1):

    Hypothesis          a falsifiable, evidence-grounded proposal
    CandidateExperiment the controlled change that tests it
    ExperimentPlan      how baseline and candidates are paired
    ComparisonReport    what the paired comparison decided

These two contracts are deliberately separate: an execution result is not a
hypothesis, and a ``FailureReport.hypothesis`` string is NOT a Hypothesis.

Design rules
------------
* Engine-agnostic. ``engine`` / ``stage`` / ``split`` / ``domain`` are plain
  strings, not Literals bound to SciNet++ or robotics, so a finance or AI4S
  adapter needs no schema change.
* Three identifiers are NEVER conflated:
    - ``experiment_id`` : one scientific experiment (may spawn several jobs)
    - ``job_id``        : one unit of execution on the engine
    - ``model_version`` : one immutable checkpoint lineage
* A failed run still yields a COMPLETE ExperimentResult. No partial dicts.
* Agents exchange these structures / file paths, never large tensors or
  console prose.
"""
from __future__ import annotations

import datetime
import math
import re
from typing import Any, ClassVar, Literal, get_args

from pydantic import BaseModel, Field, field_validator, model_validator

Status = Literal[
    "queued", "running", "succeeded", "failed", "cancelled", "incomplete"
]
TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled", "incomplete"})


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _clean_metrics_dict(value: Any) -> dict[str, float]:
    """Keep only finite numbers. Shared by validators and from_cli_result."""
    if not isinstance(value, dict):
        return {}
    clean: dict[str, float] = {}
    for key, item in value.items():
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            continue
        number = float(item)
        if math.isfinite(number):
            clean[str(key)] = number
    return clean


class ResearchObjective(BaseModel):
    objective_id: str = Field(..., min_length=1)
    domain: str = "science"          # free-form: robotics / finance / ai4s / ...
    description: str = ""
    success_criteria: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_now)


class ExperimentSpec(BaseModel):
    experiment_id: str = Field(..., min_length=1)
    objective_id: str = Field(..., min_length=1)
    hypothesis_id: str | None = None

    # Engine-agnostic routing
    engine: str = "scinet++"
    stage: str = "train"             # train | evaluate | adapt | analyze | ...
    model_name: str = "world_model"
    model_version: str | None = None
    config_path: str = Field(..., min_length=1)
    dataset_version: str | None = None

    seed: int = 42
    split: str = "train"
    budget: dict[str, Any] = Field(default_factory=dict)

    # evaluate MUST reuse the train job's run_id (that's where the ckpt lives)
    run_id: str | None = None
    parent_run_id: str | None = None
    parent_job_id: str | None = None
    created_at: str = Field(default_factory=_now)

    def with_run_id(self, run_id: str, *, stage: str | None = None,
                    parent_job_id: str | None = None,
                    split: str | None = None) -> "ExperimentSpec":
        """Return a copy wired to a parent run (used for train -> evaluate).

        ``split`` must be set explicitly when switching to an evaluation stage:
        a train spec's ``split="train"`` would otherwise make the engine
        evaluate on the training data. The orchestrator defaults to the frozen
        test split; candidate selection passes ``validation``.
        """
        update: dict[str, Any] = {
            "run_id": run_id,
            "parent_run_id": run_id,
            "parent_job_id": parent_job_id,
        }
        if stage:
            update["stage"] = stage
        if split is not None:
            update["split"] = split
        return self.model_copy(update=update)

    def to_engine_spec(self) -> dict[str, Any]:
        """The subset handed to the engine CLI (no LangGraph-specific fields)."""
        return {
            "experiment_id": self.experiment_id,
            "objective_id": self.objective_id,
            "hypothesis_id": self.hypothesis_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "dataset_version": self.dataset_version,
            "seed": self.seed,
            "split": self.split,
            "budget": self.budget,
        }


class ExperimentResult(BaseModel):
    experiment_id: str
    job_id: str
    status: Status
    engine: str = "unknown"
    stage: str | None = None

    metrics: dict[str, float] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    checkpoint: dict[str, Any] | None = None

    error_type: str | None = None
    error: str | None = None
    run_id: str | None = None
    model_version: str | None = None
    dataset_version: str | None = None
    # Which split the engine ACTUALLY evaluated on. Never infer it from the
    # requested spec alone -- the engine's own evidence is authoritative.
    evaluation_split: str | None = None
    result_path: str | None = None
    log_path: str | None = None
    exit_code: int | None = None
    started_at: str | None = None
    finished_at: str | None = None

    # Full engine payload retained for auditability; metrics above are the
    # cleaned, machine-comparable projection of it.
    raw: dict[str, Any] = Field(default_factory=dict)
    data_quality_notes: list[str] = Field(default_factory=list)

    @field_validator("metrics", mode="before")
    @classmethod
    def _clean_metrics(cls, value: Any) -> dict[str, float]:
        return _clean_metrics_dict(value)

    @property
    def succeeded(self) -> bool:
        return self.status == "succeeded"

    @classmethod
    def from_cli_result(cls, raw: dict[str, Any], *,
                        experiment_id: str | None = None) -> "ExperimentResult":
        """Convert an engine ``result.json`` into the shared contract.

        Never raises on a malformed/failed payload: the whole point is that
        failures are first-class, fully recorded research evidence.
        """
        raw = raw or {}
        raw_metrics = raw.get("metrics") or {}
        cleaned = _clean_metrics_dict(raw_metrics)
        notes = []
        for key, value in raw_metrics.items():
            if str(key) not in cleaned:
                notes.append(f"dropped non-finite/non-numeric metric: {key}={value!r}")

        status = raw.get("status") or "incomplete"
        if status not in get_args(Status):
            notes.append(f"unknown engine status {status!r} -> 'incomplete'")
            status = "incomplete"

        return cls(
            experiment_id=experiment_id or raw.get("experiment_id") or "unknown",
            job_id=str(raw.get("job_id") or "unknown"),
            status=status,
            engine=raw.get("engine") or "unknown",
            stage=raw.get("stage"),
            metrics=cleaned,
            artifacts=list(raw.get("artifacts") or []),
            checkpoint=raw.get("checkpoint"),
            error_type=raw.get("error_type"),
            error=raw.get("error"),
            run_id=raw.get("run_id"),
            model_version=raw.get("model_version"),
            dataset_version=raw.get("dataset_version"),
            evaluation_split=raw.get("evaluation_split"),
            result_path=raw.get("result_path"),
            log_path=raw.get("log_path"),
            exit_code=raw.get("exit_code"),
            started_at=raw.get("started_at"),
            finished_at=raw.get("finished_at"),
            raw=raw,
            data_quality_notes=notes,
        )

    @classmethod
    def incomplete(cls, *, experiment_id: str, job_id: str,
                  engine: str = "unknown", error: str,
                  error_type: str = "result_unavailable") -> "ExperimentResult":
        """Synthesise a complete record when the engine never returned one."""
        return cls(
            experiment_id=experiment_id, job_id=job_id, status="incomplete",
            engine=engine, error=error, error_type=error_type,
            finished_at=_now(),
        )


class FailureReport(BaseModel):
    failure_id: str = Field(default_factory=lambda: "fr_" + datetime.datetime.now()
                            .strftime("%Y%m%d%H%M%S%f"))
    experiment_id: str
    job_id: str
    layer: Literal["runtime", "model"] = "model"
    category: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    hypothesis: str | None = None
    recommended_action: str
    severity: Literal["low", "medium", "high"] = "medium"
    created_at: str = Field(default_factory=_now)


class ModelVersion(BaseModel):
    """Immutable lineage record -- you must be able to answer 'from what?'."""
    version_id: str = Field(..., min_length=1)
    experiment_id: str
    job_id: str
    engine: str = "unknown"

    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None
    config_path: str | None = None
    config_sha256: str | None = None
    # Reproducibility needs BOTH repos: the model engine AND the Agent runtime
    # that orchestrated the run. A dirty worktree is not a reproducible commit.
    code_version: str | None = None            # engine repo HEAD
    runtime_code_version: str | None = None   # Agent runtime repo HEAD
    code_dirty: bool | None = None            # None = could not be determined
    dataset_version: str | None = None
    parent_version_id: str | None = None

    metrics: dict[str, float] = Field(default_factory=dict)
    promoted: bool = False
    promotion_reason: str | None = None
    created_at: str = Field(default_factory=_now)

    @field_validator("metrics", mode="before")
    @classmethod
    def _clean_metrics(cls, value: Any) -> dict[str, float]:
        return _clean_metrics_dict(value)


# --------------------------------------------------------------------------- #
# P1: research decision protocol
# --------------------------------------------------------------------------- #
# None of these structures knows what a "world model", "predictor" or
# "pendulum" is. Metric names, config paths and split names are plain strings;
# only the *scientific* invariants are enforced here (evidence, falsifiability,
# paired seeds, held-out selection, frozen final-eval split).

CONFIG_KEY_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_-]*(?:\.[A-Za-z_][A-Za-z0-9_-]*)*$"
)
# Splits reserved for FINAL independent confirmation. Candidate selection must
# never evaluate on them. A domain may override the set with its own names.
RESERVED_FINAL_SPLITS = frozenset({"test"})


class MetricPrediction(BaseModel):
    """What the hypothesis expects to happen to ONE named metric."""
    metric: str = Field(..., min_length=1)
    direction: Literal["decrease", "increase", "unchanged"]
    expected_change_pct: float | None = None


class Hypothesis(BaseModel):
    """A machine-checkable research proposal, NOT a natural-language string.

    Invariants that make it usable by an experiment loop:
      * at least one evidence reference (no free-floating speculation);
      * at least one intervention (a concrete, reviewable change);
      * at least one predicted effect AND one observable falsification criterion.
    Passing these checks proves the hypothesis is *well formed*, not that its
    proposed mechanism is scientifically true -- that requires an experiment.
    """
    hypothesis_id: str = Field(..., min_length=1)
    objective_id: str = Field(..., min_length=1)
    source_experiment_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    statement: str = Field(..., min_length=1)
    mechanism: str = Field(..., min_length=1)
    evidence_refs: list[str] = Field(..., min_length=1)
    intervention: dict[str, Any] = Field(..., min_length=1)
    predicted_effects: list[MetricPrediction] = Field(..., min_length=1)
    falsification_criteria: list[str] = Field(..., min_length=1)
    confounders: list[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    status: Literal["proposed", "approved", "rejected", "tested"] = "proposed"
    created_at: str = Field(default_factory=_now)

    @field_validator("evidence_refs", "falsification_criteria")
    @classmethod
    def _non_empty_entries(cls, value: list[str]) -> list[str]:
        if any(not str(item).strip() for item in value):
            raise ValueError("entries must be non-empty strings")
        return value

    @field_validator("intervention")
    @classmethod
    def _valid_config_paths(cls, value: dict[str, Any]) -> dict[str, Any]:
        for key in value:
            if not isinstance(key, str) or not CONFIG_KEY_RE.match(key):
                raise ValueError(
                    f"intervention key {key!r} must be a dotted config path"
                )
        return value


class CandidateExperiment(BaseModel):
    """One controlled change, repeated over paired seeds on a held-out split."""
    candidate_id: str = Field(..., min_length=1)
    hypothesis_id: str = Field(..., min_length=1)
    config_overrides: dict[str, Any] = Field(..., min_length=1)
    seeds: list[int] = Field(..., min_length=1)
    dataset_version: str = Field(..., min_length=1)
    train_split: str = "train"
    # Deliberately a str, not Literal["validation"]: other domains call their
    # selection split "dev" or "holdout". What is enforced is that the FINAL
    # confirmation split (normally "test") can NEVER be used for selection.
    evaluation_split: str = "validation"
    epochs: int = Field(default=10, ge=1)
    timeout_s: int = Field(default=3600, ge=1)

    reserved_final_splits: ClassVar[frozenset[str]] = RESERVED_FINAL_SPLITS

    @model_validator(mode="after")
    def validate_candidate(self) -> "CandidateExperiment":
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if self.train_split == self.evaluation_split:
            raise ValueError("train_split and evaluation_split must differ")
        if self.evaluation_split in self.reserved_final_splits:
            raise ValueError(
                f"evaluation_split={self.evaluation_split!r} is reserved for "
                "final confirmation; selection must use a held-out split"
            )
        for key in self.config_overrides:
            if not isinstance(key, str) or not CONFIG_KEY_RE.match(key):
                raise ValueError(
                    f"config_overrides key {key!r} must be a dotted config path"
                )
        return self


class ExperimentPlan(BaseModel):
    """Baseline + candidates on ONE dataset, compared over EXACTLY paired seeds."""
    plan_id: str = Field(..., min_length=1)
    objective_id: str = Field(..., min_length=1)
    baseline_experiment_id: str = Field(..., min_length=1)
    dataset_version: str = Field(..., min_length=1)
    candidates: list[CandidateExperiment] = Field(..., min_length=1, max_length=2)
    primary_metric: str = Field(..., min_length=1)
    metric_direction: Literal["minimize", "maximize"] = "minimize"
    paired_seeds: list[int] = Field(..., min_length=1)
    # Guardrail rules are domain policy, not protocol constants. Values were
    # widened from ``dict[str, float]`` to ``dict[str, Any]`` in P1-05 because
    # a RELATIVE regression rule must declare its own metric direction (a
    # string); guessing direction from a metric name is explicitly forbidden.
    # Older ``{"metric": {"min": 0.8}}`` rules remain valid unchanged.
    guardrails: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # Concurrency is a DEPLOYMENT policy for the campaign runner, not a data
    # protocol constant (a 4 GB GPU here, a cluster elsewhere). The protocol
    # only guarantees >=1 and defaults to serial execution.
    max_parallel: int = Field(default=1, ge=1)
    created_at: str = Field(default_factory=_now)

    @model_validator(mode="after")
    def validate_plan(self) -> "ExperimentPlan":
        if len(set(self.paired_seeds)) != len(self.paired_seeds):
            raise ValueError("paired_seeds must be unique")
        ids = [candidate.candidate_id for candidate in self.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("candidate_id must be unique within a plan")
        for candidate in self.candidates:
            if candidate.dataset_version != self.dataset_version:
                raise ValueError("all candidates must use the same dataset_version")
            if candidate.seeds != self.paired_seeds:
                raise ValueError("all candidates must use exactly paired_seeds")
        return self


class ComparisonReport(BaseModel):
    """The paired-comparison decision contract (P1-01 protocol, P1-05 payload).

    ``accepted_for_confirmation`` is NOT promotion: it only means the candidate
    earned a final independent evaluation on the reserved split. Promotion
    belongs to P1-06 and never happens in this module.

    P1-05 extends this class BACKWARD-COMPATIBLY: every new field has a
    default, so all P1-01 callers keep constructing valid reports. The
    original validator (equal, NON-EMPTY arrays) is deliberately untouched --
    ``comparison.schemas.PairedComparisonReport`` is the subclass that relaxes
    it for the "zero valid pairs" case, where the decision must be
    ``inconclusive``.
    """
    comparison_id: str = Field(..., min_length=1)
    plan_id: str = Field(..., min_length=1)
    candidate_id: str = Field(..., min_length=1)
    primary_metric: str = Field(..., min_length=1)
    baseline_values: list[float]
    candidate_values: list[float]
    paired_deltas: list[float]
    mean_delta: float
    median_delta: float
    confidence_interval_95: tuple[float, float] | None = None
    guardrail_violations: list[str] = Field(default_factory=list)
    decision: Literal["accepted_for_confirmation", "rejected", "inconclusive"]
    notes: list[str] = Field(default_factory=list)

    # ---- P1-05 additions (all optional => no breakage) -------------------
    campaign_id: str = ""
    metric_direction: Literal["minimize", "maximize"] = "minimize"
    delta_convention: Literal["positive_is_better"] = "positive_is_better"
    n_planned: int = 0
    n_pairs: int = 0
    std_delta: float | None = None
    interval_method: str | None = None
    confidence_level: float = 0.95
    # The interval actually used for the decision. It equals ci95 for one
    # candidate; with Holm correction it is WIDER, so the ci95 field is never
    # mislabelled with a non-95% interval.
    decision_confidence_interval: tuple[float, float] | None = None
    practical_effect_threshold: float = 0.0
    practical_effect_mode: Literal["absolute", "relative"] = "absolute"
    decision_rule: str | None = None
    multiple_comparison: dict[str, Any] = Field(default_factory=dict)
    guardrail_results: list[dict[str, Any]] = Field(default_factory=list)
    missing_pairs: list[dict[str, Any]] = Field(default_factory=list)
    paired_observations: list[dict[str, Any]] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    comparison_algorithm_version: str = ""
    baseline_config_sha256: str | None = None
    candidate_config_sha256: str | None = None
    dataset_version: str | None = None
    evaluation_split: str | None = None
    dataset_split_hashes: dict[str, str] | None = None
    report_sha256: str | None = None
    created_at: str = Field(default_factory=_now)

    @model_validator(mode="after")
    def validate_pairs(self) -> "ComparisonReport":
        count = len(self.baseline_values)
        if (count == 0 or len(self.candidate_values) != count
                or len(self.paired_deltas) != count):
            raise ValueError(
                "baseline_values / candidate_values / paired_deltas must be "
                "non-empty and of equal length"
            )
        return self
