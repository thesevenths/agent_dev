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

__all__ = [name for name in dir() if not name.startswith("_")]
