"""P1-06: the trusted research-state contract.

This package exists because an LLM tool argument is NOT evidence. Before P1-06
the only "approval" in the runtime was ``CampaignRunSpec.approved_hypothesis_ids``
-- a list supplied by the caller. Nothing proved that a plan was reviewed by the
deterministic Critic, that a human ever saw it, or that it was not edited after
being approved. These structures make all three claims verifiable.

Design rules
------------
* Engine-agnostic: no metric, config key, dataset or model name appears here.
* Every content-addressed resource carries ``*_sha256`` over its CANONICAL
  payload. An approval is bound to a plan HASH, not to a plan ID.
* Statuses are non-confusable: ``pending`` can never authorise an execution;
  ``approved`` can only be produced by the out-of-band console (see
  ``approval_cli.py``), which is deliberately NOT an Agent tool.
* ``ResearchSession`` links Objective / Hypotheses / Reviews / Plans / Approvals /
  Campaigns / ComparisonReports / ResearchReport for ONE objective, and caps the
  number of campaign rounds so an inconclusive result cannot loop forever.
"""
from __future__ import annotations

import datetime
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")

ApprovalStatus = Literal["pending", "approved", "rejected", "revoked"]
ApprovalSource = Literal["human", "system_policy"]
ResearchStatus = Literal[
    "open", "awaiting_approval", "running", "completed", "cancelled"
]
ResearchOutcome = Literal[
    "not_evaluated", "accepted_for_confirmation", "rejected", "inconclusive"
]

# The only capabilities an approval may grant. A plan approval does NOT grant
# model promotion or frozen-test confirmation: those are separate future gates.
APPROVAL_SCOPES = frozenset({"run_campaign", "resume_campaign"})
DEFAULT_APPROVAL_SCOPES = ("run_campaign", "resume_campaign")


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def safe_id(value: str, label: str = "identifier") -> str:
    if (not isinstance(value, str) or not _SAFE_ID_RE.match(value)
            or ".." in value):
        raise ValueError(f"{label} must be a safe single-segment id: {value!r}")
    return value


class StoredObjective(BaseModel):
    """M4.0-01 -- persisted form of ``schemas.ResearchObjective``.

    Superset of the existing ResearchObjective fields plus the relationship /
    provenance bookkeeping required by the unified knowledge model. JSON blobs
    are stored as ``*_json`` columns in the SAME research.sqlite.
    """
    objective_id: str
    domain: str = "science"
    description: str = ""
    constraints: dict[str, Any] = Field(default_factory=dict)
    success_criteria: dict[str, Any] = Field(default_factory=dict)
    parent_objective_id: str | None = None
    status: str = "proposed"          # proposed|active|closed|abandoned|reconstructed
    origin_experience_id: str | None = None
    content_sha256: str
    created_at: str = Field(default_factory=_now)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_record(self) -> "StoredObjective":
        safe_id(self.objective_id, "objective_id")
        if self.parent_objective_id:
            safe_id(self.parent_objective_id, "parent_objective_id")
        if not self.content_sha256:
            raise ValueError("content_sha256 is required")
        return self

    @staticmethod
    def _dump_json(value: dict[str, Any]) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)

    @staticmethod
    def _load_json(raw: str | None) -> dict[str, Any]:
        if not raw:
            return {}
        try:
            out = json.loads(raw)
            return out if isinstance(out, dict) else {}
        except (ValueError, TypeError):
            return {}

    @classmethod
    def from_row(cls, row: Any) -> "StoredObjective":
        return cls(
            objective_id=row["objective_id"], domain=row["domain"],
            description=row["description"],
            constraints=cls._load_json(row["constraints_json"]),
            success_criteria=cls._load_json(row["success_criteria_json"]),
            parent_objective_id=row["parent_objective_id"],
            status=row["status"],
            origin_experience_id=row["origin_experience_id"],
            content_sha256=row["content_sha256"],
            created_at=row["created_at"],
            provenance=cls._load_json(row["provenance_json"]),
        )


class StoredHypothesis(BaseModel):
    """An immutable version of a Hypothesis, addressed by its content hash."""
    hypothesis_id: str
    objective_id: str
    hypothesis: dict[str, Any]
    content_sha256: str
    version: int = 1
    source_experiment_id: str = ""
    status: str = "proposed"
    created_at: str = Field(default_factory=_now)


class CriticReviewRecord(BaseModel):
    """The REAL result of the deterministic P1-02 Critic, bound to a hash.

    A review of hypothesis content H proves nothing about a modified H'. The
    service therefore compares ``hypothesis_sha256`` against the stored version.
    """
    review_id: str
    hypothesis_id: str
    hypothesis_sha256: str
    reviewer: str = "deterministic_critic"
    passed: bool
    errors: list[str] = Field(default_factory=list)
    review_version: str = "p1-02"
    created_at: str = Field(default_factory=_now)


class StoredExperimentPlan(BaseModel):
    plan_id: str
    objective_id: str
    plan: dict[str, Any]
    plan_sha256: str
    version: int = 1
    hypothesis_ids: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_now)


class ApprovalRecord(BaseModel):
    """A HUMAN (or explicitly declared system policy) authorisation.

    ``resource_sha256`` is the whole point: edit the plan after approval and the
    hash no longer matches, so the approval cannot be replayed against a
    different experiment.
    """
    approval_id: str
    resource_type: str = "experiment_plan"
    resource_id: str
    resource_sha256: str
    status: ApprovalStatus = "pending"
    # "human" is the ONLY source that may authorise execution by default.
    source: ApprovalSource = "human"
    approved_by: str = ""
    approved_at: str | None = None
    requested_by: str = ""
    requested_at: str = Field(default_factory=_now)
    scope: list[str] = Field(default_factory=lambda: list(DEFAULT_APPROVAL_SCOPES))
    notes: str = ""
    created_at: str = Field(default_factory=_now)

    @model_validator(mode="after")
    def validate_record(self) -> "ApprovalRecord":
        safe_id(self.approval_id, "approval_id")
        if not self.resource_id or not self.resource_sha256:
            raise ValueError("approval must bind resource_id and resource_sha256")
        if self.scope and any(item not in APPROVAL_SCOPES for item in self.scope):
            raise ValueError(f"unknown approval scope; allowed: {sorted(APPROVAL_SCOPES)}")
        if self.status == "approved":
            if not self.approved_by.strip():
                raise ValueError("an approved record requires a non-empty approved_by")
            if not self.approved_at:
                raise ValueError("an approved record requires approved_at")
        return self


class ResearchSession(BaseModel):
    """One objective's complete research context (no execution logic)."""
    session_id: str
    objective_id: str
    objective: dict[str, Any] = Field(default_factory=dict)
    status: ResearchStatus = "open"
    max_campaign_rounds: int = Field(default=2, ge=1)
    hypothesis_ids: list[str] = Field(default_factory=list)
    review_ids: list[str] = Field(default_factory=list)
    plan_ids: list[str] = Field(default_factory=list)
    approval_ids: list[str] = Field(default_factory=list)
    campaign_ids: list[str] = Field(default_factory=list)
    comparison_ids: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    report_id: str | None = None
    report_path: str | None = None
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)

    @model_validator(mode="after")
    def validate_session(self) -> "ResearchSession":
        safe_id(self.session_id, "session_id")
        if not self.objective_id:
            raise ValueError("objective_id is required")
        return self

    def link(self, field: str, values: list[str]) -> None:
        """Append unique ids, preserving history order."""
        current = getattr(self, field)
        for value in values:
            if value not in current:
                current.append(value)
        self.updated_at = _now()


class CampaignBinding(BaseModel):
    """Idempotency key: (plan_id, approval_id, plan_sha256) -> ONE campaign."""
    campaign_id: str
    plan_id: str
    approval_id: str
    plan_sha256: str
    objective_id: str
    session_id: str | None = None
    created_at: str = Field(default_factory=_now)


class ResearchReport(BaseModel):
    """Final review of a session: structured evidence, never LLM prose.

    ``promotion`` is intentionally a constant. ``accepted_for_confirmation`` is a
    scientific verdict, NOT a promotion; the frozen final split is a separate,
    human-gated phase that this release deliberately does not expose as a tool.
    """
    report_id: str
    session_id: str
    objective_id: str
    outcome: ResearchOutcome = "not_evaluated"
    campaign_count: int = 0
    comparison_count: int = 0
    hypotheses: list[dict[str, Any]] = Field(default_factory=list)
    reviews: list[dict[str, Any]] = Field(default_factory=list)
    plans: list[dict[str, Any]] = Field(default_factory=list)
    approvals: list[dict[str, Any]] = Field(default_factory=list)
    campaigns: list[dict[str, Any]] = Field(default_factory=list)
    comparisons: list[dict[str, Any]] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    experience_refs: list[str] = Field(default_factory=list)
    promotion: Literal["not_requested"] = "not_requested"
    frozen_test_used: bool = False
    decision_rules: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_now)
    report_sha256: str | None = None
