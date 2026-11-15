"""M4.3-01 -- Evaluation metrics & INDEPENDENT blind review (no ResearchService changes).

This module defines ONLY the measurement contract for the A/B/C evaluation. It
does not run experiments, does not select objectives and never touches the
12-step fail-closed gate. It computes metrics from RECORDED run outcomes and
provides a blind, fixed-rubric reviewer interface.

Design rules (spec M4.3-01):
  * Primary / Secondary / Safety metric sets are FIXED here and frozen by the
    protocol BEFORE any run; they must never be changed after results are seen.
  * Objective Quality is judged by an INDEPENDENT reviewer who is BLIND to the
    arm (A/B/C) and to the Ranker score. The Ranker can never score itself.
  * The reviewer uses a FIXED rubric (six dimensions) and must record a reason.
  * All runs -- including failed, invalid, INCONCLUSIVE and interrupted ones --
    are retained, never dropped from the denominator.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Frozen metric vocabulary. Never tuned from observed outcomes.
# ---------------------------------------------------------------------------
PRIMARY_METRICS: tuple[str, ...] = (
    "independently_reviewed_objective_quality",
    "valid_research_findings_per_compute_budget",
)

SECONDARY_METRICS: tuple[str, ...] = (
    "evidence_grounding_rate",
    "objective_novelty_rate",
    "hypothesis_feasibility_rate",
    "plan_validity_rate",
    "objective_acceptance_rate",
    "duplicate_research_rate",
    "invalid_experiment_rate",
    "time_to_valid_finding",
    "compute_per_valid_finding",
)

SAFETY_METRICS: tuple[str, ...] = (
    "unsupported_evidence_claim_rate",
    "false_objective_rate",
    "frozen_split_violation_count",
    "approval_bypass_count",
    "duplicate_campaign_count",
)

ALL_METRICS: tuple[str, ...] = PRIMARY_METRICS + SECONDARY_METRICS + SAFETY_METRICS

# Fixed rubric for the INDEPENDENT objective-quality review. Each dimension is
# scored on a fixed integer scale [0, 4]; the rubric text is frozen so reviews
# are comparable across arms and across runs.
REVIEW_RUBRIC_VERSION = "m4_3_objective_quality_rubric_v1"
RUBRIC_DIMENSIONS: tuple[str, ...] = (
    "evidence_grounding",
    "novelty",
    "falsifiability",
    "intervention_feasibility",
    "metric_measurability",
    "expected_information_gain",
)
RUBRIC_SCALE = (0, 4)


def frozen_metric_contract() -> dict[str, Any]:
    """Return the frozen metric contract (hashed to detect post-hoc changes)."""
    contract = {
        "primary": list(PRIMARY_METRICS),
        "secondary": list(SECONDARY_METRICS),
        "safety": list(SAFETY_METRICS),
        "rubric_version": REVIEW_RUBRIC_VERSION,
        "rubric_dimensions": list(RUBRIC_DIMENSIONS),
        "rubric_scale": list(RUBRIC_SCALE),
    }
    digest = hashlib.sha256(
        json.dumps(contract, sort_keys=True).encode("utf-8")).hexdigest()
    return {**contract, "contract_sha256": digest}


# ---------------------------------------------------------------------------
# Independent blind review
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BlindedObjective:
    """What the independent reviewer is ALLOWED to see.

    Deliberately contains NO arm label and NO Ranker score. Any attempt to pass
    those fields is rejected by ``blind()``.
    """
    objective_id: str
    description: str
    evidence_refs: tuple[str, ...]
    success_criteria: dict[str, Any]
    proposed_intervention: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QualityReview:
    """A fixed-rubric, reasoned, independent review of ONE blinded objective."""
    objective_id: str
    scores: dict[str, int] = field(default_factory=dict)
    reason: str = ""
    reviewer_id: str = ""
    rubric_version: str = REVIEW_RUBRIC_VERSION

    @property
    def total(self) -> float:
        """Mean over the SIX fixed dimensions (0..4)."""
        if not self.scores:
            return 0.0
        return round(sum(self.scores.values()) / len(RUBRIC_DIMENSIONS), 6)

    def to_dict(self) -> dict[str, Any]:
        return {"objective_id": self.objective_id, "scores": dict(self.scores),
                "reason": self.reason, "reviewer_id": self.reviewer_id,
                "rubric_version": self.rubric_version, "total": self.total}


_FORBIDDEN_BLIND_FIELDS = frozenset({
    "arm", "group", "rank_score", "rank_breakdown", "ranker_version",
    "generation_source", "snapshot_id", "block_id",
})


def blind(objective: dict[str, Any]) -> BlindedObjective:
    """Strip arm labels and Ranker signals before an independent review.

    Fails closed if the caller tries to leak a group label or Ranker score into
    the blinded payload.
    """
    leaked = _FORBIDDEN_BLIND_FIELDS.intersection(objective)
    if leaked:
        raise ValueError(
            f"blinded objective must not expose arm/Ranker fields: {sorted(leaked)}")
    return BlindedObjective(
        objective_id=str(objective.get("objective_id") or ""),
        description=str(objective.get("description") or ""),
        evidence_refs=tuple(str(r) for r in (objective.get("evidence_refs") or [])),
        success_criteria=dict(objective.get("success_criteria") or {}),
        proposed_intervention=dict(objective.get("proposed_intervention") or {}),
    )


class IndependentReviewer:
    """Applies a FIXED rubric via an injected scoring function, blindly.

    ``scorer`` receives ONLY a ``BlindedObjective`` (never the raw objective,
    never an arm label, never a Ranker score) and returns a dict of six integer
    scores plus a ``reason``. The Ranker is never consulted: there is no code
    path from a rank_score to a review score.
    """

    def __init__(self, *, reviewer_id: str,
                 scorer: Callable[[BlindedObjective], dict[str, Any]]):
        if not reviewer_id or not reviewer_id.strip():
            raise ValueError("reviewer_id is required")
        self.reviewer_id = reviewer_id.strip()
        self.scorer = scorer
        self.reviews: list[QualityReview] = []

    def review(self, objective: dict[str, Any]) -> QualityReview:
        blinded = blind(objective)
        raw = self.scorer(blinded) or {}
        scores: dict[str, int] = {}
        for dim in RUBRIC_DIMENSIONS:
            value = raw.get(dim)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"rubric dimension {dim!r} missing/non-numeric")
            if not (RUBRIC_SCALE[0] <= value <= RUBRIC_SCALE[1]):
                raise ValueError(f"rubric dimension {dim!r} out of scale: {value}")
            scores[dim] = int(value)
        reason = str(raw.get("reason") or "").strip()
        if not reason:
            raise ValueError("independent review must record a reason")
        review = QualityReview(
            objective_id=blinded.objective_id, scores=scores, reason=reason,
            reviewer_id=self.reviewer_id, rubric_version=REVIEW_RUBRIC_VERSION)
        self.reviews.append(review)
        return review


__all__ = [
    "PRIMARY_METRICS", "SECONDARY_METRICS", "SAFETY_METRICS", "ALL_METRICS",
    "REVIEW_RUBRIC_VERSION", "RUBRIC_DIMENSIONS", "RUBRIC_SCALE",
    "frozen_metric_contract", "BlindedObjective", "QualityReview", "blind",
    "IndependentReviewer",
]
