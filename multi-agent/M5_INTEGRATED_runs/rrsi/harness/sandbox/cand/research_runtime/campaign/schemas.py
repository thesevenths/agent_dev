"""P1-04: campaign state protocol, kept SEPARATE from the P1 decision contract.

``ExperimentPlan`` (P1-01) says WHAT should be compared. These structures say
WHAT ACTUALLY HAPPENED while executing it. Keeping them apart means:

* approving a plan never mutates execution state;
* a failed trial is a first-class record, not an exception that loses evidence;
* a process restart can rebuild the whole campaign from persisted state,
  including the exact ExperimentSpec that was submitted for each job.

No class here names an engine, a metric, a model field or a dataset format.
"""
from __future__ import annotations

import datetime
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

CampaignStatus = Literal[
    "created", "preflighting", "running", "paused",
    "completed", "completed_with_failures", "failed", "cancelled",
]
TrialStatus = Literal[
    "pending", "config_ready", "train_submitted", "training",
    "train_succeeded", "eval_submitted", "evaluating",
    "succeeded", "failed", "skipped",
]

CAMPAIGN_TERMINAL_STATUSES = frozenset({
    "completed", "completed_with_failures", "failed", "cancelled",
})
TRIAL_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "skipped"})

# A campaign_id becomes a directory name. Reject separators and traversal
# here as well as in the config builder: two independent boundaries.
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


class CampaignRunSpec(BaseModel):
    """Deployment parameters for ONE execution of an approved plan.

    Deliberately not part of ExperimentPlan: the same scientific plan may be
    run on a laptop (serial, 1 seed) or a cluster (parallel, 5 seeds) without
    changing what is being tested.
    """
    campaign_id: str = Field(..., min_length=1)
    objective_id: str = Field(..., min_length=1)
    plan_id: str = Field(..., min_length=1)
    baseline_config_path: str = Field(..., min_length=1)
    dataset_version: str = Field(..., min_length=1)
    evaluation_split: str = "validation"
    # P1-04 is serial by construction. A value > 1 is rejected in preflight
    # rather than silently ignored (that would fake a capability we lack).
    max_parallel: int = Field(default=1, ge=1)
    job_timeout_s: float = Field(default=3600.0, ge=1)
    max_jobs: int = Field(default=100, ge=1)
    max_total_seconds: float = Field(default=86400.0, ge=1)
    resume: bool = False
    # P1-02 approved these hypotheses; the campaign re-checks that every
    # candidate actually points at one of them (no unreviewed intervention).
    approved_hypothesis_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_spec(self) -> "CampaignRunSpec":
        if not _SAFE_ID_RE.match(self.campaign_id) or ".." in self.campaign_id:
            raise ValueError("campaign_id must be a safe single-segment identifier")
        return self


class TrialRecord(BaseModel):
    """One arm (baseline OR candidate) on ONE seed: train -> validation.

    The full ``train_spec`` / ``evaluation_spec`` are persisted because
    ``ExperimentManager._specs`` is process-local memory. After a restart the
    only correct way to collect an old job is ``collect_result(job_id, spec)``.
    """
    trial_id: str = Field(..., min_length=1)
    campaign_id: str
    arm_type: Literal["baseline", "candidate"]
    arm_id: str
    seed: int
    hypothesis_id: str | None = None

    config_path: str
    config_relpath: str | None = None
    config_sha256: str
    baseline_config_sha256: str | None = None
    dataset_version: str

    train_job_id: str | None = None
    train_run_id: str | None = None
    evaluation_job_id: str | None = None
    evaluation_split: str | None = None
    primary_metric: str | None = None
    status: TrialStatus = "pending"

    train_metrics: dict[str, float] = Field(default_factory=dict)
    validation_metrics: dict[str, float] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)

    # Persisted specs, NOT inferred from a live ExperimentManager.
    train_spec: dict[str, Any] | None = None
    evaluation_spec: dict[str, Any] | None = None

    train_result_path: str | None = None
    evaluation_result_path: str | None = None
    log_path: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    skip_reason: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


class CampaignRecord(BaseModel):
    """The complete, reloadable state of one campaign."""
    campaign_id: str
    plan_id: str
    objective_id: str
    status: CampaignStatus = "created"
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)
    dataset_version: str
    baseline_config_sha256: str | None = None
    evaluation_split: str = "validation"

    trials: list[TrialRecord] = Field(default_factory=list)

    # Serialized inputs: resume() must work with NO caller-supplied plan.
    plan: dict[str, Any] = Field(default_factory=dict)
    run_spec: dict[str, Any] = Field(default_factory=dict)
    policy_snapshot: dict[str, Any] = Field(default_factory=dict)

    event_log: list[str] = Field(default_factory=list)
    final_manifest_path: str | None = None

    @property
    def completed_trials(self) -> int:
        return sum(t.status == "succeeded" for t in self.trials)

    @property
    def failed_trials(self) -> int:
        return sum(t.status == "failed" for t in self.trials)

    @property
    def skipped_trials(self) -> int:
        return sum(t.status == "skipped" for t in self.trials)

    def trial(self, trial_id: str) -> TrialRecord | None:
        return next((t for t in self.trials if t.trial_id == trial_id), None)
