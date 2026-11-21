"""M6 -- failure-driven Research Harness candidate generation.

Pipeline (exactly the M6 contract):
  failure classification -> bottleneck detection -> historical experience
  retrieval -> improvement proposal -> candidate generation -> static
  validation -> candidate freeze.

The generator receives ONLY dev-split tasks and failures from dev/selection
*executions already released as evidence*. It never receives holdout task ids,
labels, metrics or gold. The boundary is enforced, not documented: any payload
that mentions a holdout task id, a sealed ref, or a gold field is rejected.

Candidates are *research methods* (e.g. "a hypothesis is accepted only if it
states a falsifying observation"), not prompt rewrites. Every candidate retains
its parent SHA, source experience ids, failure evidence, expected improvement
and falsification conditions. Unsuccessful candidates are kept by the caller.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .benchmark import BenchmarkSpec, ContaminationLedger
from .harness import (
    HarnessBundle, HarnessError, assert_single_method_change, derive_candidate,
    freeze_bundle,
)

FAILURE_TO_COMPONENT = {
    "EVIDENCE_GROUNDING_WEAK": "objective_generator",
    "NOVELTY_DUPLICATE": "objective_generator",
    "HYPOTHESIS_UNFALSIFIABLE": "hypothesis_critic",
    "CRITERION_VAGUE": "hypothesis_critic",
    "CONFOUNDER_UNCONTROLLED": "experiment_designer",
    "NO_CONTROL_ARM": "experiment_designer",
    "ROOT_CAUSE_MISATTRIBUTED": "failure_diagnoser",
    "TRANSFER_NOT_ASSESSED": "failure_diagnoser",
}

# Fixed, pre-registered method improvements. These are NOT tuned from holdout
# results; the library is part of the candidate generator's frozen version.
IMPROVEMENT_LIBRARY: dict[str, dict[str, Any]] = {
    "objective_generator": {
        "version": "method_v2_evidence_novelty_rank",
        "config": {"min_evidence_refs": 2, "require_novelty_check": True,
                   "require_research_value": True, "rank_by": "value_times_evidence"},
        "expected": "Select fewer but better-grounded, non-duplicate research questions",
        "falsification": ["objective_quality on dev does not improve",
                          "duplicate objective rate does not fall"],
    },
    "hypothesis_critic": {
        "version": "method_v2_falsifiability_gate",
        "config": {"require_falsifying_observation": True,
                   "require_negative_predictor": True,
                   "require_evidence_binding": True, "min_criterion_precision": 2},
        "expected": "Reject unfalsifiable hypotheses before spending campaign budget",
        "falsification": ["falsifiability rate does not improve on dev",
                          "valid finding rate falls"],
    },
    "experiment_designer": {
        "version": "method_v2_confounder_gate",
        "config": {"require_control_arm": True, "require_confounder_list": True,
                   "require_paired_seeds": True, "require_preregistered_metric": True},
        "expected": "Reduce invalid experiments caused by confounding or missing controls",
        "falsification": ["experiment validity does not improve on dev",
                          "budget efficiency regresses"],
    },
    "failure_diagnoser": {
        "version": "method_v2_root_cause_chain",
        "config": {"require_evidence_chain": True, "require_counterfactual": True,
                   "require_experience_link": True, "require_transfer_check": True},
        "expected": "Identify root causes with traceable evidence and transfer notes",
        "falsification": ["diagnosis accuracy does not improve on dev",
                          "transfer performance does not improve"],
    },
}

_FORBIDDEN_TOKENS = re.compile(
    r"(gold_ref|rubric_ref|holdout_task|expected_direction|minimum_meaningful_effect)",
    re.IGNORECASE)


class GenerationBoundaryViolation(ValueError):
    """Raised when a generator payload leaks sealed evaluation information."""


class EvidenceBoundary:
    """Hard boundary around what a candidate generator may observe."""

    def __init__(self, benchmark: BenchmarkSpec, ledger: ContaminationLedger):
        self.benchmark = benchmark
        self.ledger = ledger
        self.allowed_task_ids = {t.task_id for t in benchmark.by_split("dev")}
        self.holdout_task_ids = {t.task_id for t in benchmark.by_split("holdout")}
        self.selection_task_ids = {t.task_id for t in benchmark.by_split("selection")}

    def assert_allowed(self, payload: Any) -> dict[str, Any]:
        blob = str(payload)
        if _FORBIDDEN_TOKENS.search(blob):
            raise GenerationBoundaryViolation("sealed evaluation field in generator payload")
        ids = set(re.findall(r"m6_bench_v1_[a-z]+_[a-z]+_\d{3}", blob))
        leaked = ids & (self.holdout_task_ids | self.selection_task_ids)
        if leaked:
            raise GenerationBoundaryViolation(
                f"non-dev task visible to generator: {sorted(leaked)}")
        return {"ok": True, "n_allowed": len(self.allowed_task_ids)}

    def grant(self, *, harness_sha: str, candidate_sha: str) -> dict[str, Any]:
        return self.ledger.grant(harness_sha=harness_sha,
                                 candidate_sha=candidate_sha,
                                 visible_task_ids=sorted(self.allowed_task_ids),
                                 phase="candidate_generation")


@dataclass
class FailureRecord:
    task_id: str
    category: str
    evidence: str
    experience_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "category": self.category,
                "evidence": self.evidence, "experience_id": self.experience_id}


def classify_failures(records: list[dict[str, Any]]) -> list[FailureRecord]:
    """Map raw failure text to the fixed taxonomy (deterministic keywords)."""
    rules = [
        ("falsifi", "HYPOTHESIS_UNFALSIFIABLE"),
        ("vague criterion", "CRITERION_VAGUE"),
        ("confound", "CONFOUNDER_UNCONTROLLED"),
        ("control arm", "NO_CONTROL_ARM"),
        ("root cause", "ROOT_CAUSE_MISATTRIBUTED"),
        ("transfer", "TRANSFER_NOT_ASSESSED"),
        ("duplicate", "NOVELTY_DUPLICATE"),
        ("evidence", "EVIDENCE_GROUNDING_WEAK"),
    ]
    out: list[FailureRecord] = []
    for r in records:
        text = f"{r.get('reason','')} {r.get('detail','')}".lower()
        category = next((c for k, c in rules if k in text), "EVIDENCE_GROUNDING_WEAK")
        out.append(FailureRecord(
            task_id=str(r.get("task_id", "")), category=category,
            evidence=str(r.get("evidence", r.get("detail", ""))),
            experience_id=str(r.get("experience_id", ""))))
    return out


@dataclass
class CandidateRecord:
    candidate_id: str
    parent_sha: str
    candidate_sha: str
    changed_component: str
    source_experience_ids: list[str] = field(default_factory=list)
    failure_evidence: list[dict[str, Any]] = field(default_factory=list)
    expected_improvement: str = ""
    falsification_conditions: list[str] = field(default_factory=list)
    creation_provenance: dict[str, Any] = field(default_factory=dict)
    artifact_path: str = ""
    file_sha256: str = ""
    status: str = "FROZEN"

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


def detect_bottleneck(failures: list[FailureRecord]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for f in failures:
        counts[f.category] = counts.get(f.category, 0) + 1
    if not counts:
        return {"category": "EVIDENCE_GROUNDING_WEAK", "n_failures": 0,
                "counts": {}, "component": "objective_generator"}
    dominant = max(sorted(counts), key=lambda k: counts[k])
    return {"category": dominant, "n_failures": len(failures),
            "counts": counts, "component": FAILURE_TO_COMPONENT[dominant]}


def _retrieve_experience(failures: list[FailureRecord],
                         boundary: EvidenceBoundary) -> list[str]:
    """Historical experience ids, restricted to boundary-visible tasks."""
    ids = {f.experience_id for f in failures
           if f.experience_id and f.task_id in boundary.allowed_task_ids}
    return sorted(ids)


def generate_candidate(*, base: HarnessBundle, failures: list[dict[str, Any]],
                       benchmark: BenchmarkSpec, ledger: ContaminationLedger,
                       out_dir: str | Path, candidate_id: str,
                       component: str | None = None) -> dict[str, Any]:
    """Run the full failure-driven generation pipeline and freeze ONE candidate."""
    boundary = EvidenceBoundary(benchmark, ledger)
    boundary.assert_allowed(failures)
    classified = classify_failures(failures)
    bottleneck = detect_bottleneck(classified)
    chosen = component or bottleneck["component"]
    improvement = IMPROVEMENT_LIBRARY[chosen]
    exp_ids = _retrieve_experience(classified, boundary)

    cand = derive_candidate(
        base, component=chosen, version=improvement["version"],
        config=improvement["config"], candidate_id=candidate_id,
        source_experience_ids=exp_ids,
        expected_improvement=improvement["expected"],
        falsification_conditions=improvement["falsification"])
    change = assert_single_method_change(base, cand)

    frozen = freeze_bundle(cand, Path(out_dir) / f"{candidate_id}.json")
    record = CandidateRecord(
        candidate_id=candidate_id, parent_sha=base.sha256,
        candidate_sha=frozen["candidate_sha"], changed_component=chosen,
        source_experience_ids=exp_ids,
        failure_evidence=[f.to_dict() for f in classified],
        expected_improvement=improvement["expected"],
        falsification_conditions=improvement["falsification"],
        creation_provenance={"bottleneck": bottleneck,
                             "boundary": boundary.assert_allowed(failures),
                             "improvement_library_version": "m6_improvements_v1"},
        artifact_path=frozen["path"], file_sha256=frozen["file_sha256"])
    grant = boundary.grant(harness_sha=base.sha256,
                           candidate_sha=record.candidate_sha)
    return {"candidate": cand, "record": record, "bottleneck": bottleneck,
            "change": change, "generation_view": benchmark.generator_view(),
            "contamination_grant": grant}


__all__ = [
    "GenerationBoundaryViolation", "EvidenceBoundary", "FailureRecord",
    "CandidateRecord", "FAILURE_TO_COMPONENT", "classify_failures",
    "detect_bottleneck", "generate_candidate",
]
