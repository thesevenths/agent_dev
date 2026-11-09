"""P1-05 Paired Comparison layer: EVALUATE campaign facts, never execute them.

    CampaignStore / CampaignRecord (P1-04 facts)
        -> CampaignComparator  (P1-05B) pairs, validates, decides
        -> ComparisonReportStore (P1-05A) immutable, hash-traceable report

Strict separation of duties (do not blur these in later phases):

    CampaignRunner   EXECUTES            (P1-04)
    Comparator       EVALUATES           (P1-05)
    ModelRegistry    PROMOTES            (P1-06, NOT here)
    CodeAgent        ORCHESTRATES        (P1-06, NOT here)

``accepted_for_confirmation`` is a scientific verdict, not a promotion. No
module in this package imports ModelRegistry, tools.py, an LLM or an engine.
"""
from .exceptions import (
    ComparisonError,
    ComparisonInputError,
    ComparisonIntegrityError,
    ComparisonStoreError,
)
from .guardrails import GuardrailEvaluator
from .comparator import CampaignComparator
from .statistics import (
    INTERVAL_METHOD,
    HolmEntry,
    HolmResult,
    PairedStatistics,
    PairedStatisticsEngine,
    holm_adjust,
    normalize_deltas,
    student_t_critical,
    student_t_two_sided_p,
)
from .report_store import (
    ComparisonReportStore,
    ReportArtifact,
    canonical_report_payload,
    content_sha256,
    sha256_bytes,
)
from .schemas import (
    COMPARISON_ALGORITHM_VERSION,
    ComparisonPolicy,
    GuardrailAssessment,
    GuardrailResult,
    MissingPair,
    PairedComparisonReport,
    PairedObservation,
)

__all__ = [
    "COMPARISON_ALGORITHM_VERSION", "INTERVAL_METHOD",
    "ComparisonPolicy", "PairedObservation", "MissingPair",
    "GuardrailResult", "GuardrailAssessment", "PairedComparisonReport",
    "CampaignComparator", "GuardrailEvaluator",
    "PairedStatisticsEngine", "PairedStatistics", "HolmResult", "HolmEntry",
    "normalize_deltas", "holm_adjust", "student_t_critical",
    "student_t_two_sided_p",
    "ComparisonReportStore", "ReportArtifact",
    "content_sha256", "canonical_report_payload", "sha256_bytes",
    "ComparisonError", "ComparisonIntegrityError",
    "ComparisonInputError", "ComparisonStoreError",
]
