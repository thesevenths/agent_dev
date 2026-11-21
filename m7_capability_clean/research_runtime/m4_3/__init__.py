"""M4.3-01 / -01.1 / -02 -- A/B/C value-validation runtime (ADDITIVE package).

Nothing here modifies, forks or re-implements ResearchService, the Planner, the
Evidence Reader, the Decision Engine, the 12-step gate or any RRSI component.

Layers
------
* ``evaluation_harness`` / ``experiment_registry`` / ``metrics`` /
  ``blind_review`` / ``evidence_boundary`` / ``research_service_gate`` /
  ``protocol_schema``  -- M4.3-01(.1) measurement + gate evidence.
* ``experiment_state`` / ``experiment_controller`` / ``experiment_recovery``
  -- M4.3-02 unified stage/budget/recovery coordination.
* ``review_queue``         -- durable HUMAN blind-review submission entry point.
* ``pilot_runner``         -- REAL SciNet++ A/B/C pilot execution.
* ``pilot_analysis`` / ``sample_size`` / ``statistical_analysis``
  -- descriptive pilot analysis, power/sample size, confirmatory statistics.
* ``confirmatory_protocol_builder`` / ``confirmatory_runner``
  -- freeze gate + human-gated confirmatory execution.
* ``value_report``         -- M4.3-04 pilot/final value reports.
* ``cli``                  -- the unified command line.

The engine-dependent modules (``pilot_runner``, ``confirmatory_runner``) are
NOT imported here: importing the package must stay cheap and engine-free.
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
    ALL_METRICS, METRIC_CATEGORIES, METRIC_DEFINITIONS,
    METRIC_FORMULA_VERSION, MetricDefinition, PRIMARY_METRICS,
    SAFETY_METRICS, SECONDARY_METRICS, compute_metric_set,
    frozen_metric_contract, metric_definitions,
)
from .research_service_gate import (
    ASSOCIATION_CHECKS, CAMPAIGN_SUCCESS_STATUSES, GATE_STEP_COUNT,
    GATED_OUTCOMES, SCIENTIFIC_REPORT_OUTCOMES, UNGATED_OUTCOMES,
    FORGED_GATE_KEYS, GateAdapter, GateEvidence, GateRunRef,
    ResearchServiceGateAdapter,
)
from .protocol_schema import (
    CONFIRMATORY_REQUIRED_FIELDS, POSTHOC_LOCKED_FIELDS, EMPTY_VALUES,
    ProtocolImmutable, ProtocolNotFrozen, ProtocolRegistry, ProtocolVersion,
    assert_confirmatory_ready, missing_confirmatory_fields,
)
from .experiment_state import (
    ACTIVE_STATES, STATES, TERMINAL_STATES, ExperimentEventStore,
    ExperimentRecord, ExperimentStateMachine, IllegalTransition,
)
from .experiment_controller import (
    STAGE_CONFIRMATORY, STAGE_PILOT, BudgetExhausted, BudgetGuard,
    ExperimentController,
)
from .experiment_recovery import (
    RecoveryReport, recover_experiment, reconcile,
)
from .review_queue import QueueItem, ReviewQueue, ReviewQueueViolation
from .pilot_analysis import analyze_pilot, render_pilot_report
from .sample_size import ContrastSpec, estimate_sample_size
from .statistical_analysis import (
    analyse_primary_contrasts, paired_block_contrast,
)
from .value_report import VALUE_OUTCOMES, generate_value_report

__all__ = [
    # harness / registry
    "EvaluationHarness", "ProtocolViolation", "ArmContext", "BlockSpec",
    "build_arm_context", "ExperimentRegistry", "FrozenProtocol", "RunRecord",
    "ARMS",
    # blind review
    "IndependentReviewer", "QualityReview", "ReviewPanel",
    "BlindedObjective", "BlindingEntry", "BlindingLedger", "BlindingViolation",
    "blind", "FORBIDDEN_BLIND_FIELDS",
    "REVIEW_RUBRIC_VERSION", "RUBRIC_DIMENSIONS", "RUBRIC_SCALE",
    "REVIEW_KINDS", "REVIEW_KIND_HUMAN_BLIND", "REVIEW_KIND_AUTOMATED",
    # metrics
    "frozen_metric_contract", "PRIMARY_METRICS", "SECONDARY_METRICS",
    "SAFETY_METRICS", "ALL_METRICS", "METRIC_CATEGORIES",
    "METRIC_DEFINITIONS", "METRIC_FORMULA_VERSION", "MetricDefinition",
    "compute_metric_set", "metric_definitions",
    # gate evidence
    "GATE_STEP_COUNT", "GATED_OUTCOMES", "UNGATED_OUTCOMES",
    "FORGED_GATE_KEYS", "CAMPAIGN_SUCCESS_STATUSES",
    "SCIENTIFIC_REPORT_OUTCOMES", "ASSOCIATION_CHECKS", "GateAdapter",
    "GateEvidence", "GateRunRef", "ResearchServiceGateAdapter",
    # protocol
    "CONFIRMATORY_REQUIRED_FIELDS", "POSTHOC_LOCKED_FIELDS", "EMPTY_VALUES",
    "ProtocolImmutable", "ProtocolNotFrozen", "ProtocolRegistry",
    "ProtocolVersion", "assert_confirmatory_ready",
    "missing_confirmatory_fields",
    # controller / state / recovery
    "STATES", "ACTIVE_STATES", "TERMINAL_STATES", "IllegalTransition",
    "ExperimentStateMachine", "ExperimentRecord", "ExperimentEventStore",
    "ExperimentController", "BudgetGuard", "BudgetExhausted",
    "STAGE_PILOT", "STAGE_CONFIRMATORY", "RecoveryReport",
    "recover_experiment", "reconcile",
    # review queue
    "ReviewQueue", "ReviewQueueViolation", "QueueItem",
    # analysis / reporting
    "analyze_pilot", "render_pilot_report", "estimate_sample_size",
    "ContrastSpec", "paired_block_contrast", "analyse_primary_contrasts",
    "generate_value_report", "VALUE_OUTCOMES",
]
