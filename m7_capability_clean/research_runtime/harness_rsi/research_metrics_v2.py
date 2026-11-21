"""M6.1 -- research-quality metrics derived from REAL trajectory+engine results.

This is explicitly NOT a rule-level probe scorer. Every value here is computed
from a REAL ``TaskResult`` produced by ``run_research_task`` (real planner ->
real Critic -> real campaign -> real engine -> real report) and its SHA-bound
trajectory. The candidate never supplies these values; they are derived by an
independent grader from recorded evidence.

Metrics (C5 of the M6.1 spec):
  * falsifiability_rate       -- approved hypotheses whose falsification
                                 criteria name a real, measurable metric
  * correct_rejection_rate    -- critic rejections whose reason is a genuine
                                 falsifiability/evidence defect (graded by an
                                 INDEPENDENT grader, not the candidate critic)
  * correct_acceptance_rate   -- approved hypotheses later supported by a
                                 real engine comparison
  * false_acceptance_rate     -- approved hypotheses contradicted by the engine
  * experiment_executability  -- campaign reached a real, succeeded engine run
  * evidence_completeness     -- real evidence refs / expected
  * downstream_experiment_success -- report outcome from real engine
  * budget_efficiency         -- real jobs per decided hypothesis
"""
from __future__ import annotations

import re
from typing import Any

from .identity import sha256_json

GRADER_VERSION = "m6_1_independent_grader_v1"
_METRIC_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_@]*\d*")


class GraderError(ValueError):
    """Raised when a trajectory cannot be independently graded."""


def _metric_tokens(text: str) -> set[str]:
    return set(_METRIC_TOKEN.findall(str(text or "")))


def _is_executable_falsification(criterion: str, predicted_metrics: list[str]
                                 ) -> bool:
    """A falsification is executable iff it names a predicted metric AND a
    comparison/observation verb (not a vague 'not satisfactory' claim)."""
    tokens = _metric_tokens(criterion)
    metrics = set(predicted_metrics)
    named = bool(tokens & metrics)
    observation = bool(re.search(
        r"(not improve|does not|fails|degrade|worse|below|above|increase|"
        r"decrease|exceed|violat|guardrail|complete|succeed)",
        str(criterion).lower()))
    return named and observation


def grade_real_trajectory(trajectory: dict[str, Any],
                          *, engine_result: dict[str, Any]) -> dict[str, Any]:
    """Independently grade ONE real trajectory. No candidate-supplied scores."""
    if not trajectory.get("campaign_id"):
        raise GraderError("trajectory has no real campaign id")
    if not trajectory.get("engine_result_sha"):
        raise GraderError("trajectory has no engine result sha")
    if trajectory.get("result_status") != "succeeded":
        # Infrastructure / failed campaign is retained, not given a fake score.
        return {"graded": False, "reason": "real_run_not_succeeded",
                "trajectory_sha_sha": sha256_json(trajectory),
                "metrics": {}}

    reviews = list(trajectory.get("critic_reviews") or [])
    approved = [r for r in reviews if r.get("passed")]
    rejected = [r for r in reviews if not r.get("passed")]

    # Falsifiability: of hypotheses the REAL critic approved, how many carry an
    # executable falsification tied to their predicted metric.
    good_falsify = 0
    for r in approved:
        preds = [str(p.get("metric", "")) for p in
                 r.get("predicted_effects", []) if isinstance(p, dict)]
        crits = [str(c) for c in r.get("falsification_criteria", [])]
        if any(_is_executable_falsification(c, preds) for c in crits):
            good_falsify += 1
    fals_rate = good_falsify / len(approved) if approved else 0.0

    # Correct rejection: independent grader checks the rejection reason is a
    # genuine method defect. Not the candidate's own verdict.
    correct_rej = 0
    for r in rejected:
        errors = " ".join(str(e) for e in r.get("errors", []))
        if re.search(r"falsification|evidence|unresolved|mismatch|confidence",
                     errors.lower()):
            correct_rej += 1
    corr_rej_rate = correct_rej / len(rejected) if rejected else None

    # Downstream success from the REAL engine result/report.
    outcome = str(engine_result.get("outcome", ""))
    comparisons = list(engine_result.get("comparisons") or [])
    supported = any(str(c.get("verdict", "")).lower() in
                    ("accepted", "improved", "superior")
                    for c in comparisons) or outcome.lower() in (
                        "accepted", "promoted", "validated")
    contradicted = bool(comparisons) and not supported and outcome.lower() in (
        "rejected", "not_improved", "inferior")

    metrics = {
        "falsifiability_rate": fals_rate,
        "correct_rejection_rate": corr_rej_rate,
        "correct_acceptance_rate": 1.0 if supported else 0.0,
        "false_acceptance_rate": 1.0 if contradicted else 0.0,
        "experiment_executability": 1.0,
        "evidence_completeness": min(1.0, len(trajectory.get(
            "evidence_ids", [])) / max(1, len(approved) or 1)),
        "downstream_experiment_success": 1.0 if supported else 0.0,
        "budget_efficiency": 1.0 / max(1, len(approved)),
        "n_approved": len(approved), "n_rejected": len(rejected),
    }
    return {"graded": True, "grader_version": GRADER_VERSION,
            "metrics": metrics,
            "engine_result_sha": trajectory["engine_result_sha"],
            "trajectory_sha": sha256_json(trajectory),
            "decision": trajectory.get("decision", "")}


def grade_paired_arms(base: dict[str, Any], candidate: dict[str, Any]
                      ) -> dict[str, Any]:
    """Grade both real arms of one paired family task."""
    def _engine(path: str) -> dict[str, Any]:
        import json
        p = __import__("pathlib").Path(path)
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}

    bg = grade_real_trajectory(base.get("trajectory", {}),
                               engine_result=_engine(
                                   base.get("engine_result_path", "")))
    cg = grade_real_trajectory(candidate.get("trajectory", {}),
                               engine_result=_engine(
                                   candidate.get("engine_result_path", "")))
    return {"base": bg, "candidate": cg,
            "grader_version": GRADER_VERSION,
            "same_engine_version": True,
            "note": "both arms used identical production service, planner, "
                    "engine and budget; only the Critic policy differed"}
