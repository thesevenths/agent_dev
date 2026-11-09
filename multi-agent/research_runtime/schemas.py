"""
research_runtime.schemas
========================
The single data contract between the Agent Runtime and ANY scientific engine.

Four questions, four structures:

    ResearchObjective   what are we actually trying to prove?
    ExperimentSpec      what exactly was run?
    ExperimentResult    what came out (success OR failure)?
    FailureReport       why did it fail / what should be tested next?
    ModelVersion        which checkpoint, which data, which parent, promoted?

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
from typing import Any, Literal, get_args

from pydantic import BaseModel, Field, field_validator

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
                    parent_job_id: str | None = None) -> "ExperimentSpec":
        """Return a copy wired to a parent run (used for train -> evaluate)."""
        update: dict[str, Any] = {
            "run_id": run_id,
            "parent_run_id": run_id,
            "parent_job_id": parent_job_id,
        }
        if stage:
            update["stage"] = stage
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
    code_version: str | None = None
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
