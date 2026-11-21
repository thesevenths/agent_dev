"""ARR 1.0 / F1 -- Research Failure Analyzer.

Reads REAL artefacts (``TaskResult`` / ``ResearchReport`` / experiment results /
comparison reports / telemetry / holdout + OOD measurements) and identifies
where the RESEARCH PROCESS itself failed. This is deliberately different from
``research_runtime.failure_diagnoser``, which diagnoses a scientific MODEL's
failure from engine metrics. Here the subject is the Research Harness.

Every finding carries measured evidence (numbers/ids from the run), never a
free-form opinion. Categories map 1:1 to the Harness knobs in
``evolution_spaces.HARNESS_SEARCH_SPACE`` so a diagnosis can produce an
actionable, controlled candidate.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable

FAILURE_CATEGORIES = (
    "HYPOTHESIS_QUALITY_FAILURE",
    "EXPERIMENT_DESIGN_FAILURE",
    "TOOL_USAGE_FAILURE",
    "PLANNING_FAILURE",
    "WORKFLOW_FAILURE",
    "EVALUATION_FAILURE",
    "MEMORY_FAILURE",
    "COST_FAILURE",
)


@dataclass
class FailureFinding:
    category: str
    severity: str                    # low | medium | high
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)
    suggested_target: str = ""
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        blob = json.dumps({"category": self.category, "target": self.suggested_target},
                          sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class FailureAnalysis:
    task_id: str = ""
    domain: str = ""
    findings: list[FailureFinding] = field(default_factory=list)
    dominant: str = ""
    quality_Q: float | None = None
    source_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id, "domain": self.domain,
            "findings": [f.to_dict() for f in self.findings],
            "dominant": self.dominant, "quality_Q": self.quality_Q,
            "source_sha256": self.source_sha256,
        }


def _checks(result: dict[str, Any]) -> dict[str, bool]:
    return dict(((result.get("quality") or {}).get("checks")) or {})


def analyze_task_result(result: dict[str, Any] | Any) -> FailureAnalysis:
    """Diagnose the RESEARCH PROCESS from one real TaskResult."""
    if hasattr(result, "model_dump"):
        data = result.model_dump(mode="json")
    elif isinstance(result, dict):
        data = result
    else:
        raise TypeError("result must be a TaskResult or its dict payload")
    report = data.get("report") or {}
    quality = data.get("quality") or {}
    telemetry = data.get("telemetry") or {}
    evidence = data.get("evidence") or {}
    harness = (data.get("provenance") or {}).get("harness") or {}
    comparisons = data.get("comparisons") or []
    hypotheses = data.get("hypotheses") or []
    chk = _checks(data)
    decisions = [str(c.get("decision")) for c in comparisons]
    findings: list[FailureFinding] = []

    # 1) Hypothesis quality: few / ungrounded / critic-rejected proposals.
    if (len(hypotheses) < 2
            or not chk.get("D.hypotheses_are_evidence_grounded", False)
            or telemetry.get("critic_rejections")):
        findings.append(FailureFinding(
            "HYPOTHESIS_QUALITY_FAILURE", "high",
            "too few or insufficiently evidence-grounded hypotheses",
            {"n_hypotheses": len(hypotheses),
             "evidence_grounded": chk.get("D.hypotheses_are_evidence_grounded"),
             "critic_rejections": telemetry.get("critic_rejections", []),
             "proposer_skip_reasons": telemetry.get("proposer_skip_reasons", [])},
            "planner.hypothesis_prompt", 0.9))

    # 2) Experiment design: cannot discriminate competing hypotheses.
    if (not chk.get("D.competing_interventions_tested", False)
            or decisions.count("inconclusive") > 0
            or harness.get("planner_strategy") == "repeat_refuted"):
        findings.append(FailureFinding(
            "EXPERIMENT_DESIGN_FAILURE", "high",
            "experiments did not discriminate competing hypotheses",
            {"n_candidates": len(comparisons), "decisions": decisions,
             "planner_strategy": harness.get("planner_strategy"),
             "n_inconclusive": decisions.count("inconclusive")},
            "planner.strategy", 0.9))

    # 3) Tool/engine usage: real trial failures / missing runs.
    missing = int(telemetry.get("missing_runs") or 0)
    timeouts = int(telemetry.get("timeout_runs") or 0)
    if missing or timeouts or not data.get("ok"):
        findings.append(FailureFinding(
            "TOOL_USAGE_FAILURE", "high",
            "engine/tool runs did not complete",
            {"ok": data.get("ok"), "missing_runs": missing,
             "timeout_runs": timeouts, "error": data.get("error", "")},
            "planner.strategy", 0.95))

    # 4) Planning: no breadth, no competing interventions.
    if (harness.get("max_candidates", 1) < 2
            or not chk.get("D.competing_interventions_tested", False)):
        findings.append(FailureFinding(
            "PLANNING_FAILURE", "medium",
            "planner explored only one candidate",
            {"max_candidates": harness.get("max_candidates"),
             "n_hypotheses": len(hypotheses)},
            "workflow.max_candidates", 0.8))

    # 5) Workflow: campaign rounds / next-round capability.
    if harness.get("campaign_rounds", 1) < 2:
        findings.append(FailureFinding(
            "WORKFLOW_FAILURE", "medium",
            "single-round workflow cannot iterate on evidence",
            {"campaign_rounds": harness.get("campaign_rounds"),
             "campaigns": len(report.get("campaigns") or [])},
            "workflow.campaign_rounds", 0.7))

    # 6) Evaluation: holdout / OOD / hard constraint problems.
    holdout = evidence.get("holdout") or {}
    ood = evidence.get("ood") or {}
    if (not holdout.get("evaluated")
            or not chk.get("G.holdout_confirmed_improvement", False)
            or quality.get("hard_violations")):
        findings.append(FailureFinding(
            "EVALUATION_FAILURE", "high",
            "finding not confirmed on an independent holdout / hard constraints",
            {"holdout_evaluated": holdout.get("evaluated"),
             "mean_holdout_delta": holdout.get("mean_holdout_delta"),
             "ood_mean_delta": ood.get("mean_holdout_delta"),
             "hard_violations": quality.get("hard_violations", [])},
            "reviewer.require_falsification", 0.85))

    # 7) Memory: no retrieval of past (including refuted) experience.
    if harness.get("experience_top_k", 0) <= 0:
        findings.append(FailureFinding(
            "MEMORY_FAILURE", "medium",
            "planner ignores historical successes and refutations",
            {"experience_top_k": harness.get("experience_top_k")},
            "memory.experience_top_k", 0.8))

    # 8) Cost: wasted trials / budget overrun.
    total = int(telemetry.get("total_trials") or 0)
    succeeded = int(telemetry.get("succeeded_trials") or 0)
    elapsed = float(telemetry.get("elapsed_seconds") or 0.0)
    budget_seconds = float((telemetry.get("budget") or {}).get(
        "max_total_seconds") or 0.0)
    if (total and succeeded < total) or (budget_seconds
                                          and elapsed > budget_seconds):
        findings.append(FailureFinding(
            "COST_FAILURE", "medium", "wasted trials or budget overrun",
            {"total_trials": total, "succeeded_trials": succeeded,
             "elapsed_seconds": elapsed, "budget_seconds": budget_seconds},
            "workflow.campaign_rounds", 0.9))

    severity_rank = {"high": 3, "medium": 2, "low": 1}
    findings.sort(key=lambda f: (-severity_rank.get(f.severity, 0),
                                 -f.confidence))
    dominant = findings[0].category if findings else ""
    source = json.dumps(data, sort_keys=True, ensure_ascii=False,
                        default=str).encode("utf-8")
    return FailureAnalysis(
        task_id=str(data.get("task_id") or ""),
        domain=str(data.get("domain") or ""),
        findings=findings, dominant=dominant,
        quality_Q=(quality.get("Q") if isinstance(quality.get("Q"),
                                                  (int, float)) else None),
        source_sha256=hashlib.sha256(source).hexdigest(),
    )


def analyze_many(results: Iterable[dict[str, Any] | Any]) -> dict[str, Any]:
    """Aggregate findings across REAL runs; the dominant failure is the one
    with the highest weighted severity count (not an LLM judgement)."""
    analyses = [analyze_task_result(r) for r in results]
    counts: dict[str, int] = {c: 0 for c in FAILURE_CATEGORIES}
    for analysis in analyses:
        for finding in analysis.findings:
            counts[finding.category] = counts.get(finding.category, 0) + 1
    dominant = max(counts, key=lambda c: counts[c]) if analyses else ""
    return {
        "n_tasks": len(analyses),
        "counts": counts,
        "dominant": dominant,
        "analyses": [a.to_dict() for a in analyses],
        "source_sha256": hashlib.sha256(
            json.dumps([a.source_sha256 for a in analyses],
                       sort_keys=True).encode("utf-8")).hexdigest(),
    }
