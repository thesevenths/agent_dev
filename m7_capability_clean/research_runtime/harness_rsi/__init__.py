"""M6 -- unified Research Harness Recursive Self-Improvement subsystem.

This package is a NATURAL EXTENSION of the M5 runtime, not a second framework:
it reuses RuntimeState/RuntimeCoordinator, RRSICoordinator/EvolutionRegistry,
the M5 structured holdout verifier, and the M4.3 frozen statistics/blind-review
assets. It adds only the missing research-method layer:

  * versioned research-method components + immutable Harness bundles
  * a sealed, family-disjoint research benchmark
  * failure-driven candidate generation with a hard evidence boundary
  * controlled paired Base/Candidate execution and independent statistics
  * formal fail-closed promotion, real adoption and real rollback
  * four kinds of Experience and a retrieval ablation
"""
from .identity import (IdentityError, freeze_artifact, freeze_json,
                       git_fingerprint, is_sha256, sha256_bytes, sha256_file,
                       sha256_json, verify_frozen)
from .benchmark import (BENCHMARK_VERSION, BenchmarkError, BenchmarkSpec,
                        ContaminationLedger, TaskSpec, build_benchmark,
                        build_tasks, pilot_power_analysis, validate_benchmark)
from .harness import (BASE_HARNESS_ID, RESEARCH_COMPONENTS, ComponentSpec,
                      HarnessBundle, HarnessError, assert_single_method_change,
                      base_harness, derive_candidate, freeze_bundle,
                      load_frozen_bundle)
from .candidate_generation import (CandidateRecord, EvidenceBoundary,
                                   FailureRecord, GenerationBoundaryViolation,
                                   classify_failures, detect_bottleneck,
                                   generate_candidate)
from .method_runner import deterministic_method_runner
from .real_engine import build_real_runner
from .execution import ExecutionError, classify_failure, run_paired
from .evaluation import (EVALUATOR_VERSION, METRICS, EvaluationError,
                         PreRegistration, TaskResult, analyse_paired,
                         compute_metric_set, freeze_preregistration)
from .promotion import (GATE_VERSION, PromotionError, formal_promote,
                        gate_version_sha, make_artifact_reader,
                        verify_formal_promotion)
from .adoption import ADOPTION_VERSION, AdoptionError, AdoptionManager
from .rollback import RollbackError, rollback
from .experience import (EXPERIENCE_KINDS, ExperienceBank, ExperienceError,
                         ResearchExperience, run_retrieval_ablation)

# M6.1 Acceptance Closure -- Benchmark V2, real trajectories, family stats,
# formal loader/dispatcher binding and runtime-safe rollback.
from . import benchmark_v2, critic_strategies, real_trajectory, statistics_v2
from . import harness_loader, rollback_v2
from .benchmark_v2 import (
    BENCHMARK_VERSION as BENCHMARK_V2_VERSION, BenchmarkIsolationError,
    HoldoutAccessVerifier, build_benchmark_v2, validate_benchmark_v2,
    assert_no_holdout_in_generator_view,
)
from .critic_strategies import (
    CRITIC_VERSION, build_critic_manager_factory, strategy_from_bundle,
)
from .statistics_v2 import (
    STAT_VERSION, FamilyResult, PreregistrationV2,
    analyse_family_confirmation, freeze_preregistration_v2,
)
from .harness_loader import (
    LOADER_VERSION, CampaignVersionBinding, HarnessLoader, LoaderError,
    LoadedHarness,
)
from .rollback_v2 import ROLLBACK_V2_VERSION, real_rollback

from .confirmatory import (
    CONFIRM_VERSION, ConfirmationError, freeze_candidate_for_selection,
    run_confirmatory_holdout,
)
__all__ = [name for name in dir() if not name.startswith("_")]
