"""M4.3-01 / M4.3-01.1 -- Evaluation metrics contract.

This module defines ONLY the measurement contract for the A/B/C evaluation.
It does not run experiments, does not select objectives and never touches the
12-step fail-closed gate. It computes metrics from RECORDED run outcomes.

The blind independent review implementation lives in ``blind_review.py``
(P0-02: opaque review_id, sealed mapping, >= 2 reviewers, append-only
reviews, human-blind vs automated separation). It is re-exported here so all
existing ``from research_runtime.m4_3.metrics import ...`` call sites keep
working against the hardened implementation.

Design rules:
  * Primary / Secondary / Safety metric sets are FIXED here and frozen by the
    protocol BEFORE any run; they must never be changed after results are seen.
  * ``review_kinds`` is part of the frozen contract: an AUTOMATED score must
    never be mixed into ``independently_reviewed_objective_quality``.
"""
from __future__ import annotations

import hashlib
import json

# Re-export the hardened blind-review contract (single source of truth).
from .blind_review import (  # noqa: F401
    FORBIDDEN_BLIND_FIELDS, REVIEW_KIND_AUTOMATED, REVIEW_KIND_HUMAN_BLIND,
    REVIEW_KINDS, REVIEW_RUBRIC_VERSION, RUBRIC_DIMENSIONS, RUBRIC_SCALE,
    BlindedObjective, BlindingEntry, BlindingLedger, BlindingViolation,
    IndependentReviewer, QualityReview, ReviewPanel, blind,
)

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


def frozen_metric_contract() -> dict[str, Any]:
    """Return the frozen metric contract (hashed to detect post-hoc changes).

    ``review_kinds`` is deliberately part of the hash: separating HUMAN BLIND
    from AUTOMATED scoring is a frozen property of the protocol, not a tunable.
    """
    contract = {
        "primary": list(PRIMARY_METRICS),
        "secondary": list(SECONDARY_METRICS),
        "safety": list(SAFETY_METRICS),
        "rubric_version": REVIEW_RUBRIC_VERSION,
        "rubric_dimensions": list(RUBRIC_DIMENSIONS),
        "rubric_scale": list(RUBRIC_SCALE),
        "review_kinds": list(REVIEW_KINDS),
    }
    digest = hashlib.sha256(
        json.dumps(contract, sort_keys=True).encode("utf-8")).hexdigest()
    return {**contract, "contract_sha256": digest}


__all__ = [
    "PRIMARY_METRICS", "SECONDARY_METRICS", "SAFETY_METRICS", "ALL_METRICS",
    "REVIEW_RUBRIC_VERSION", "RUBRIC_DIMENSIONS", "RUBRIC_SCALE",
    "REVIEW_KINDS", "REVIEW_KIND_HUMAN_BLIND", "REVIEW_KIND_AUTOMATED",
    "FORBIDDEN_BLIND_FIELDS", "frozen_metric_contract",
    "BlindedObjective", "QualityReview", "BlindingEntry", "BlindingLedger",
    "BlindingViolation", "IndependentReviewer", "ReviewPanel", "blind",
]
