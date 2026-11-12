"""ARR 1.0 / D3 -- the UNIFIED research task & result protocol.

Every domain (AI4S / Embodied / Quant) enters the SAME ResearchService through
one ``ResearchTask``. The task carries *intent* and *policy* only; it never
carries a hypothesis, a metric implementation or an engine handle. Domain
specifics live in the DomainBundle (adapter, config whitelist, evaluator,
dataset verifier).

Reuse note: ``to_objective()`` produces the existing ``ResearchObjective``
schema -- no new parallel objective type is introduced. ``TaskResult`` wraps
the existing immutable ``ResearchReport`` plus the D1 ``QualityScore``.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .research_quality import QUALITY_PROTOCOL_VERSION
from .schemas import ResearchObjective

PROTOCOL_VERSION = "arr-1.0-task-v1"

#: The six protocol objects. Added for discoverability/audit; the authoritative
#: implementations remain the existing schema/protocol modules.
PROTOCOL_OBJECTS = (
    "ResearchTask", "ResearchHypothesis(schemas.Hypothesis)",
    "ExperimentPlan(schemas.ExperimentPlan)",
    "ExperimentResult(schemas.ExperimentResult)",
    "ResearchReport(research_store.schemas.ResearchReport)",
    "ResearchQualityEvaluator(research_quality)",
)


def _task_id(prefix: str = "task") -> str:
    return prefix + "_" + uuid.uuid4().hex[:12]


def _now() -> str:
    import datetime
    return datetime.datetime.now().isoformat(timespec="seconds")


class ResearchTask(BaseModel):
    """One domain-independent research assignment.

    ``domain`` selects a DomainBundle; the generic runtime never branches on it.
    ``heldout_policy`` is fixed by the operator BEFORE the run and cannot be
    altered by a Harness candidate (F4).
    """

    task_id: str = Field(default_factory=lambda: _task_id("task"))
    domain: Literal["scinet", "embodied", "quant"] = "scinet"
    objective: str = Field(..., min_length=1)
    objective_id: str = ""
    constraints: dict[str, Any] = Field(default_factory=dict)
    dataset_refs: list[str] = Field(default_factory=list)
    baseline_ref: str | None = None
    budget: dict[str, Any] = Field(default_factory=lambda: {
        "experiments": 20, "max_jobs": 30, "max_total_seconds": 3600.0,
        "epochs": 1,
    })
    evaluation_protocol: dict[str, Any] = Field(default_factory=lambda: {
        "primary_metric": "",
        "metric_direction": "minimize",
        "decision_split": "validation",
        "reserved_final_split": "test",
    })
    heldout_policy: dict[str, Any] = Field(default_factory=lambda: {
        "independent_holdout": True,
        "holdout_split": "test",
        "ood_task": True,
        "paired_seeds": [42, 43, 44, 45, 46],
    })
    created_at: str = Field(default_factory=_now)

    @field_validator("objective_id", mode="before")
    @classmethod
    def _default_objective_id(cls, value: Any, info: Any) -> str:
        return value or ""

    @model_validator(mode="after")
    def validate_task(self) -> "ResearchTask":
        if not self.objective_id:
            # Unique per task so `session_for_objective` cannot silently reuse a
            # previous session (which would make runs non-independent).
            self.objective_id = "obj_" + self.task_id
        if self.evaluation_protocol.get("decision_split") == \
                self.evaluation_protocol.get("reserved_final_split"):
            raise ValueError("decision_split must differ from reserved_final_split")
        seeds = self.heldout_policy.get("paired_seeds") or []
        if len(set(seeds)) != len(seeds) or not seeds:
            raise ValueError("heldout_policy.paired_seeds must be non-empty and unique")
        if self.heldout_policy.get("holdout_split") in (
                self.evaluation_protocol.get("decision_split"), "train"):
            raise ValueError("holdout_split must be independent of selection/train")
        return self

    def to_objective(self) -> ResearchObjective:
        """Reuse the existing research objective schema unchanged."""
        return ResearchObjective(
            objective_id=self.objective_id, domain=self.domain,
            description=self.objective,
            success_criteria=dict(self.evaluation_protocol),
            constraints=dict(self.constraints),
        )

    def fingerprint(self) -> str:
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True,
                          ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


class TaskResult(BaseModel):
    """Uniform output of a research run, regardless of domain."""

    task_id: str
    domain: str
    objective_id: str
    session_id: str = ""
    ok: bool = False
    outcome: str = "not_evaluated"
    report: dict[str, Any] = Field(default_factory=dict)
    quality: dict[str, Any] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)
    campaign: dict[str, Any] = Field(default_factory=dict)
    hypotheses: list[dict[str, Any]] = Field(default_factory=list)
    comparisons: list[dict[str, Any]] = Field(default_factory=list)
    baseline: dict[str, Any] = Field(default_factory=dict)
    telemetry: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    protocol_version: str = PROTOCOL_VERSION
    quality_protocol_version: str = QUALITY_PROTOCOL_VERSION
    error: str = ""
    artifact_sha256: str = ""

    def canonical_payload(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        data.pop("artifact_sha256", None)
        return data

    def compute_sha256(self) -> str:
        blob = json.dumps(self.canonical_payload(), sort_keys=True,
                          ensure_ascii=False, separators=(",", ":"),
                          default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


def task_from_spec(*, domain: str, objective: str, **kwargs: Any) -> ResearchTask:
    return ResearchTask(domain=domain, objective=objective, **kwargs)


def default_tasks() -> list[ResearchTask]:
    """Three REAL minimum research tasks, one per domain (used by benchmark)."""
    return [
        ResearchTask(
            domain="scinet",
            objective="Reduce multi-step latent dynamics prediction error on the "
                      "frozen pendulum world model",
            dataset_refs=["pendulum_smoke_v1"],
            evaluation_protocol={"primary_metric": "latent_mse@1",
                                 "metric_direction": "minimize",
                                 "decision_split": "validation",
                                 "reserved_final_split": "test"},
            # SciNet++ has no separate OOD panel; its OOD check is the
            # CROSS-DOMAIN held-out task in Package F.
            heldout_policy={"independent_holdout": True, "holdout_split": "test",
                            "ood_task": False,
                            "paired_seeds": [42, 43, 44, 45, 46]},
        ),
        ResearchTask(
            domain="embodied",
            objective="Reduce long-horizon state prediction error of the simulated "
                      "robot world model without violating safety constraints",
            dataset_refs=["robot_sim_v1"],
            evaluation_protocol={"primary_metric": "state_mse@1",
                                 "metric_direction": "minimize",
                                 "decision_split": "validation",
                                 "reserved_final_split": "test"},
        ),
        ResearchTask(
            domain="quant",
            objective="Discover a robust market-neutral signal with positive "
                      "out-of-sample net-of-cost return and bounded drawdown",
            dataset_refs=["market_hist_v1"],
            evaluation_protocol={"primary_metric": "net_return_after_cost",
                                 "metric_direction": "maximize",
                                 "decision_split": "validation",
                                 "reserved_final_split": "test"},
        ),
    ]
