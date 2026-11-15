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
from .metrics import (
    ALL_METRICS, PRIMARY_METRICS, SAFETY_METRICS, SECONDARY_METRICS,
    IndependentReviewer, blind, frozen_metric_contract,
)

__all__ = [
    "EvaluationHarness", "ProtocolViolation", "ArmContext", "BlockSpec",
    "build_arm_context", "ExperimentRegistry", "FrozenProtocol", "RunRecord",
    "ARMS", "IndependentReviewer", "blind", "frozen_metric_contract",
    "PRIMARY_METRICS", "SECONDARY_METRICS", "SAFETY_METRICS", "ALL_METRICS",
]
