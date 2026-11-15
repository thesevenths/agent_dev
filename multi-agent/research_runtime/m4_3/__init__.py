"""M4.3-01 -- A/B/C Evaluation Harness (independent experiment protocol).

This package is ADDITIVE: it does NOT modify or fork ResearchService, the
Planner, the Evidence Reader, the Decision Engine, or any RRSI component. It
only records A/B/C runs, applies a blind independent review, and computes the
frozen metrics.
"""
from .evaluation_harness import (
    ArmContext, BlockSpec, EvaluationHarness, ProtocolViolation,
    build_arm_context,
)
from .experiment_registry import (
    ARMS, ExperimentRegistry, FrozenProtocol, RunRecord,
)
from .blind_review import (
    FORBIDDEN_BLIND_FIELDS, REVIEW_KIND_AUTOMATED, REVIEW_KIND_HUMAN_BLIND,
    REVIEW_KINDS, REVIEW_RUBRIC_VERSION, RUBRIC_DIMENSIONS, RUBRIC_SCALE,
    BlindedObjective, BlindingEntry, BlindingLedger, BlindingViolation,
    IndependentReviewer, QualityReview, ReviewPanel, blind,
)
from .metrics import (
    ALL_METRICS, PRIMARY_METRICS, SAFETY_METRICS, SECONDARY_METRICS,
    frozen_metric_contract,
)
from .research_service_gate import (
    GATE_STEP_COUNT, GATED_OUTCOMES, UNGATED_OUTCOMES, FORGED_GATE_KEYS,
    GateAdapter, GateEvidence, GateRunRef, ResearchServiceGateAdapter,
)
from .protocol_schema import (
    CONFIRMATORY_REQUIRED_FIELDS, POSTHOC_LOCKED_FIELDS, EMPTY_VALUES,
    ProtocolImmutable, ProtocolNotFrozen, ProtocolRegistry, ProtocolVersion,
    assert_confirmatory_ready, missing_confirmatory_fields,
)

__all__ = [
    "EvaluationHarness", "ProtocolViolation", "ArmContext", "BlockSpec",
    "build_arm_context", "ExperimentRegistry", "FrozenProtocol", "RunRecord",
    "ARMS", "IndependentReviewer", "QualityReview", "ReviewPanel",
    "BlindedObjective", "BlindingEntry", "BlindingLedger", "BlindingViolation",
    "blind", "frozen_metric_contract",
    "PRIMARY_METRICS", "SECONDARY_METRICS", "SAFETY_METRICS", "ALL_METRICS",
    "REVIEW_RUBRIC_VERSION", "RUBRIC_DIMENSIONS", "RUBRIC_SCALE",
    "REVIEW_KINDS", "REVIEW_KIND_HUMAN_BLIND", "REVIEW_KIND_AUTOMATED",
    "FORBIDDEN_BLIND_FIELDS",
    "GATE_STEP_COUNT", "GATED_OUTCOMES", "UNGATED_OUTCOMES", "FORGED_GATE_KEYS",
    "GateAdapter", "GateEvidence", "GateRunRef", "ResearchServiceGateAdapter",
    "CONFIRMATORY_REQUIRED_FIELDS", "POSTHOC_LOCKED_FIELDS", "EMPTY_VALUES",
    "ProtocolImmutable", "ProtocolNotFrozen", "ProtocolRegistry",
    "ProtocolVersion", "assert_confirmatory_ready",
    "missing_confirmatory_fields",
]
