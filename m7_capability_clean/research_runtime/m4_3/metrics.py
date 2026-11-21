"""M4.3-01 / M4.3-01.1 -- Evaluation metrics contract.

This module defines ONLY the measurement contract for the A/B/C evaluation.
It does not run experiments, does not select objectives and never touches the
12-step fail-closed gate. It computes metrics from RECORDED run outcomes and
provides the frozen definitions used by BOTH ``compute_metrics`` (overall) and
``metrics_by_arm`` (per-arm), so the two can never silently diverge (P1-01).

The blind independent review implementation lives in ``blind_review.py``
(P0-02). It is re-exported here so all existing
``from research_runtime.m4_3.metrics import ...`` call sites keep working.

Design rules:
  * Primary / Secondary / Safety metric sets are FIXED here and frozen by the
    protocol BEFORE any run; they must never be changed after results are seen.
  * ``review_kinds`` and ``formula_version`` are part of the frozen contract:
    an AUTOMATED score must never be mixed into a human-blind metric, and the
    metric FORMULAS are versioned so a formula change is detectable (P1-01 #8).
  * Every metric declares numerator / denominator / missing-value / failure
    handling. Nothing is silently dropped from a denominator.
  * NO significance conclusion is computed anywhere in this module.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Callable

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

# P1-01 #7: three DISTINCT metric families (safety is a cross-cutting tier).
METRIC_CATEGORIES: dict[str, tuple[str, ...]] = {
    "objective_quality": (
        "independently_reviewed_objective_quality", "evidence_grounding_rate",
        "objective_novelty_rate", "hypothesis_feasibility_rate",
        "plan_validity_rate", "unsupported_evidence_claim_rate",
        "false_objective_rate"),
    "research_finding_rate": (
        "valid_research_findings_per_compute_budget",
        "objective_acceptance_rate", "duplicate_research_rate",
        "invalid_experiment_rate", "frozen_split_violation_count",
        "approval_bypass_count"),
    "compute_efficiency": (
        "time_to_valid_finding", "compute_per_valid_finding",
        "duplicate_campaign_count"),
}

# P1-01 #8: formula version is part of the frozen metric contract hash.
METRIC_FORMULA_VERSION = "m4_3_metric_formula_v2"


@dataclass(frozen=True)
class MetricDefinition:
    """Frozen numerator/denominator/missing/failure rule for ONE metric."""
    name: str
    tier: str                 # primary | secondary | safety
    category: str             # objective_quality | research_finding_rate | compute_efficiency
    numerator: str
    denominator: str
    missing_value: Any        # value when the denominator is empty (None or 0.0)
    failure_policy: str       # how failed/invalid/interrupted runs are handled
    paired_by_block: bool = False   # P1-01 #3: Primary metrics are paired


def _tier(name: str) -> str:
    if name in PRIMARY_METRICS:
        return "primary"
    if name in SAFETY_METRICS:
        return "safety"
    return "secondary"


def _category(name: str) -> str:
    for cat, names in METRIC_CATEGORIES.items():
        if name in names:
            return cat
    raise ValueError(f"metric {name!r} has no category")


_RETAIN_ALL = ("failed / invalid / inconclusive / interrupted runs are ALWAYS "
               "retained in the denominator; never dropped")
_COUNT_ALL = ("count over ALL runs; missing flag defaults to False")

METRIC_DEFINITIONS: dict[str, MetricDefinition] = {
    name: MetricDefinition(
        name=name, tier=_tier(name), category=_category(name),
        numerator=num, denominator=den, missing_value=missing,
        failure_policy=failure,
        paired_by_block=(name in PRIMARY_METRICS))
    for name, num, den, missing, failure in (
        ("independently_reviewed_objective_quality",
         "sum of human-blind review totals", "count of human-blind reviewed runs",
         0.0, "runs without a human-blind review are excluded from the mean; "
              "automated scores are NEVER included"),
        ("valid_research_findings_per_compute_budget",
         "count of valid_finding runs", "total compute_units across ALL runs",
         0.0, _RETAIN_ALL),
        ("evidence_grounding_rate",
         "count of runs with non-empty evidence_refs", "count of ALL runs",
         0.0, _RETAIN_ALL),
        ("objective_novelty_rate",
         "count of runs flagged novel", "count of ALL runs",
         0.0, _RETAIN_ALL),
        ("hypothesis_feasibility_rate",
         "count of runs flagged feasible", "count of ALL runs",
         0.0, _RETAIN_ALL),
        ("plan_validity_rate",
         "count of runs with a verified 12-step gate", "count of ALL runs",
         0.0, "a failed/interrupted run has gate_steps_passed=0 -> counted as invalid"),
        ("objective_acceptance_rate",
         "count of valid_finding + negative_finding runs", "count of ALL runs",
         0.0, _RETAIN_ALL),
        ("duplicate_research_rate",
         "count of runs flagged duplicate", "count of ALL runs",
         0.0, _RETAIN_ALL),
        ("invalid_experiment_rate",
         "count of outcome=invalid runs", "count of ALL runs",
         0.0, _RETAIN_ALL),
        ("time_to_valid_finding",
         "sum wall_time_s over valid_finding runs", "count of valid_finding runs",
         None, "no valid finding -> None (missing, NOT zero)"),
        ("compute_per_valid_finding",
         "sum compute_units over valid_finding runs", "count of valid_finding runs",
         None, "no valid finding -> None (missing, NOT zero)"),
        ("unsupported_evidence_claim_rate",
         "count of unsupported_evidence_claim runs", "count of ALL runs",
         0.0, _COUNT_ALL),
        ("false_objective_rate",
         "count of false_objective runs", "count of ALL runs",
         0.0, _COUNT_ALL),
        ("frozen_split_violation_count",
         "count of frozen_split_violation runs", "count (absolute count, no denominator)",
         0, _COUNT_ALL),
        ("approval_bypass_count",
         "count of approval_bypass runs", "count (absolute count, no denominator)",
         0, _COUNT_ALL),
        ("duplicate_campaign_count",
         "count of duplicate_campaign runs", "count (absolute count, no denominator)",
         0, _COUNT_ALL),
    )
}


def metric_definitions() -> dict[str, dict[str, Any]]:
    return {name: asdict(d) for name, d in METRIC_DEFINITIONS.items()}


def compute_metric_set(runs: list[Any]) -> dict[str, Any]:
    """The SINGLE implementation of all 16 metrics, used by overall + per-arm.

    No drops: every run stays in the denominator (P1-01 #2). Missing values
    follow ``MetricDefinition.missing_value`` (None vs 0.0), never a silent
    substitution that would flatter an arm.
    """
    n = len(runs)

    def _rate(pred: Callable[[Any], bool]) -> float:
        return round(sum(1 for r in runs if pred(r)) / n, 6) if n else 0.0

    valid = [r for r in runs if r.is_valid_finding()]
    total_compute = sum(r.compute_units for r in runs)
    quality = [r.objective_quality for r in runs if r.quality_review]
    values: dict[str, Any] = {
        "independently_reviewed_objective_quality":
            round(sum(quality) / len(quality), 6) if quality else 0.0,
        "valid_research_findings_per_compute_budget":
            round(len(valid) / total_compute, 6) if total_compute > 0 else 0.0,
        "evidence_grounding_rate": _rate(lambda r: bool(r.evidence_refs)),
        "objective_novelty_rate": _rate(
            lambda r: bool(r.notes.get("novel", False))),
        "hypothesis_feasibility_rate": _rate(
            lambda r: bool(r.notes.get("feasible", False))),
        "plan_validity_rate": _rate(lambda r: r.gate_steps_passed == 12),
        "objective_acceptance_rate": _rate(
            lambda r: r.outcome in ("valid_finding", "negative_finding")),
        "duplicate_research_rate": _rate(
            lambda r: bool(r.notes.get("duplicate", False))),
        "invalid_experiment_rate": _rate(lambda r: r.outcome == "invalid"),
        "time_to_valid_finding":
            round(sum(r.wall_time_s for r in valid) / len(valid), 6)
            if valid else None,
        "compute_per_valid_finding":
            round(sum(r.compute_units for r in valid) / len(valid), 6)
            if valid else None,
        "unsupported_evidence_claim_rate": _rate(
            lambda r: r.unsupported_evidence_claim),
        "false_objective_rate": _rate(lambda r: r.false_objective),
        "frozen_split_violation_count": sum(
            1 for r in runs if r.frozen_split_violation),
        "approval_bypass_count": sum(1 for r in runs if r.approval_bypass),
        "duplicate_campaign_count": sum(1 for r in runs if r.duplicate_campaign),
    }
    assert set(values) == set(ALL_METRICS), "metric set must stay frozen"
    return values


def frozen_metric_contract() -> dict[str, Any]:
    """Return the frozen metric contract (hashed to detect post-hoc changes).

    ``formula_version`` and the full definition table are part of the hash, so
    changing ANY metric formula produces a different contract (P1-01 #8).
    """
    contract = {
        "primary": list(PRIMARY_METRICS),
        "secondary": list(SECONDARY_METRICS),
        "safety": list(SAFETY_METRICS),
        "rubric_version": REVIEW_RUBRIC_VERSION,
        "rubric_dimensions": list(RUBRIC_DIMENSIONS),
        "rubric_scale": list(RUBRIC_SCALE),
        "review_kinds": list(REVIEW_KINDS),
        "formula_version": METRIC_FORMULA_VERSION,
        "categories": {k: list(v) for k, v in METRIC_CATEGORIES.items()},
        "definitions": metric_definitions(),
    }
    digest = hashlib.sha256(
        json.dumps(contract, sort_keys=True).encode("utf-8")).hexdigest()
    return {**contract, "contract_sha256": digest}


__all__ = [
    "PRIMARY_METRICS", "SECONDARY_METRICS", "SAFETY_METRICS", "ALL_METRICS",
    "METRIC_CATEGORIES", "METRIC_FORMULA_VERSION", "MetricDefinition",
    "METRIC_DEFINITIONS", "metric_definitions", "compute_metric_set",
    "REVIEW_RUBRIC_VERSION", "RUBRIC_DIMENSIONS", "RUBRIC_SCALE",
    "REVIEW_KINDS", "REVIEW_KIND_HUMAN_BLIND", "REVIEW_KIND_AUTOMATED",
    "FORBIDDEN_BLIND_FIELDS", "frozen_metric_contract",
    "BlindedObjective", "QualityReview", "BlindingEntry", "BlindingLedger",
    "BlindingViolation", "IndependentReviewer", "ReviewPanel", "blind",
]
