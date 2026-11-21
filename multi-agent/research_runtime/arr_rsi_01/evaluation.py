"""ARR-RSI-01 -- independent evaluation and 4-valued verdict.

This module is imported ONLY by the runner's evaluation phase. It is never
imported by the Research Runtime, the Diagnosis implementation or any arm
constructor. It is the only module allowed to read sealed ground truth.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ..harness_rsi.evaluation import (
    PreRegistration,
    TaskResult as PairedTaskResult,
    analyse_paired,
)
from .arms import ArmSpec, assert_matched_budgets
from .diagnosis import REQUIRED_DIAGNOSIS_FIELDS
from .tasks import GroundTruth, PublicTask, assert_no_ground_truth, sha256_json

EVALUATOR_VERSION = "arr_rsi_01_eval_v1"
DIMENSIONS = (
    "scientific_correctness", "ood_performance", "evidence_completeness",
    "experimental_validity", "reproducibility", "budget_efficiency",
)
VERDICTS = ("PASS", "FAIL", "INCONCLUSIVE", "BLOCKED")


@dataclass(frozen=True)
class Preregistration:
    """Frozen before any arm is executed (artifact: experiment_manifest.json)."""

    evaluator_version: str = EVALUATOR_VERSION
    primary_metric: str = "arr_rsi_holdout_delta"
    direction: str = "maximize"
    minimum_meaningful_effect: float = 0.02
    alpha: float = 0.05
    min_n_tasks: int = 5
    min_dimension_score: float = 0.75
    min_correctness: float = 0.8
    freeze: dict[str, Any] = None  # type: ignore[assignment]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["preregistration_sha256"] = sha256_json(d)
        return d


def _num(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    import math
    return v if math.isfinite(v) else None


def _holdout_delta(arm_result: dict[str, Any]) -> float | None:
    evidence = arm_result.get("evidence") or {}
    for key in ("holdout", "ood"):
        node = evidence.get(key) or {}
        value = _num(node.get("mean_holdout_delta"))
        if value is not None:
            return value
    telemetry = arm_result.get("telemetry") or {}
    ood = telemetry.get("ood") or {}
    return _num(ood.get("mean_delta"))


def _diagnosis_score(task: PublicTask, truth: GroundTruth,
                     diagnosis: dict[str, Any]) -> dict[str, Any]:
    cause_ok = diagnosis.get("root_cause") == truth.ground_truth_root_cause
    chosen = diagnosis.get("proposed_intervention") or {}
    key = chosen.get("key") if isinstance(chosen, dict) else None
    intervention_ok = key == truth.correct_intervention
    return {
        "root_cause_correct": bool(cause_ok),
        "intervention_correct": bool(intervention_ok),
        "score": round((float(cause_ok) + float(intervention_ok)) / 2.0, 4),
    }


def _evidence_score(diagnosis: dict[str, Any],
                    trajectory: dict[str, Any]) -> dict[str, Any]:
    present = [f for f in REQUIRED_DIAGNOSIS_FIELDS
               if diagnosis.get(f) not in (None, "", [], {})]
    refs = diagnosis.get("evidence_refs") or []
    recorded: set[str] = set()

    def _collect(value: Any) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                recorded.add(str(k))
                _collect(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                _collect(v)
        elif isinstance(value, (str, int, float)) and str(value):
            recorded.add(str(value))

    _collect(trajectory)
    resolvable = [r for r in refs
                  if any(token in r for token in recorded)]
    score = len(present) / max(1, len(REQUIRED_DIAGNOSIS_FIELDS))
    if len(refs) >= 2:
        score = min(1.0, score + 0.1)
    if refs and not resolvable:
        score = min(score, 0.5)
    return {"fields_present": present, "n_refs": len(refs),
            "n_resolvable": len(resolvable), "score": round(float(score), 4)}


def evaluate_task(*, task: PublicTask, truth: GroundTruth,
                  arms: dict[str, ArmSpec], arm_results: dict[str, dict[str, Any]],
                  diagnoses: dict[str, dict[str, Any]],
                  trajectory: dict[str, Any],
                  prereg: Preregistration) -> dict[str, Any]:
    """Evaluate ONE task. Ground truth is used here and nowhere else."""
    diags = diagnoses.get(task.task_id) or diagnoses.get("base") or {}
    diagnosis = diags.get("base") if isinstance(diags, dict) and "base" in diags else diags
    scientific = _diagnosis_score(task, truth, diagnosis if isinstance(diagnosis, dict) else {})
    evidence = _evidence_score(diagnosis if isinstance(diagnosis, dict) else {},
                               trajectory)

    matched = {"matched": False, "signature": {}}
    try:
        matched = assert_matched_budgets(
            arms, seeds=list(task.budget.get("paired_seeds", [task.seed])))
    except ValueError:
        pass

    all_present = all(name in arm_results for name in ("base", "candidate",
                                                       "placebo", "shuffled"))
    all_ok = all(bool((arm_results.get(name) or {}).get("ok")) for name in
                 ("base", "candidate", "placebo", "shuffled"))
    infra_failures = [
        name for name, result in arm_results.items()
        if (result.get("telemetry") or {}).get("infrastructure_failures", 0)
    ]
    try:
        assert_no_ground_truth(diagnosis)
        leaked = False
    except Exception:
        leaked = True

    validity = bool(matched["matched"] and all_present and all_ok
                    and not infra_failures and not leaked)

    reproducibility = {
        "task_sha256": sha256_json(task.to_dict()),
        "frozen_dataset_version": bool(task.dataset_version),
        "seeds_recorded": bool(task.budget.get("paired_seeds")),
        "artifact_sha256": bool(all(
            (arm_results.get(name) or {}).get("artifact_sha256")
            for name in ("base", "candidate", "placebo", "shuffled"))),
    }
    repro_score = sum(bool(v) for v in reproducibility.values()) / len(reproducibility)

    candidate = arm_results.get("candidate") or {}
    jobs = sum(int((r.get("telemetry") or {}).get("jobs_used", 0) or 0)
               for r in arm_results.values())
    delta = _holdout_delta(candidate)
    quality = _num((candidate.get("quality") or {}).get("Q"))
    efficiency = (round(float(quality) / jobs, 6)
                  if quality is not None and jobs > 0 else None)

    ood_value = _holdout_delta(candidate)
    ood = {"available": ood_value is not None,
           "delta": ood_value,
           "positive": bool(ood_value is not None and ood_value > 0),
           "score": 1.0 if (ood_value is not None and ood_value > 0) else 0.0}

    dims = {
        "scientific_correctness": scientific["score"],
        "ood_performance": ood["score"],
        "evidence_completeness": evidence["score"],
        "experimental_validity": 1.0 if validity else 0.0,
        "reproducibility": round(repro_score, 4),
        "budget_efficiency": efficiency,
    }
    return {
        "task_id": task.task_id,
        "task_family": task.task_family,
        "dimensions": dims,
        "scientific_correctness": scientific,
        "ood": ood,
        "evidence": evidence,
        "experimental_validity": {
            "valid": validity, "matched": matched, "all_arms_present": all_present,
            "all_arms_ok": all_ok, "infrastructure_failures": infra_failures,
            "ground_truth_leak": leaked,
        },
        "reproducibility_detail": reproducibility,
        "budget": {"jobs": jobs, "quality": quality,
                   "efficiency": efficiency},
        "ground_truth_used_only_here": True,
    }


def analyse_statistics(*, task_evaluations: list[dict[str, Any]],
                       arm_results: dict[str, dict[str, dict[str, Any]]],
                       prereg: Preregistration) -> dict[str, Any]:
    """Paired candidate-vs-control analysis over ALL tasks, reuse M6 stats."""
    metrics = prereg.primary_metric
    pre = PreRegistration(
        experiment_id="arr_rsi_01",
        primary_metric=metrics,
        direction=prereg.direction,
        minimum_meaningful_effect=prereg.minimum_meaningful_effect,
        alpha=prereg.alpha,
        secondary_metrics=(),
    )
    families = {e["task_id"]: e["task_family"] for e in task_evaluations}

    def _results(arm: str) -> list[PairedTaskResult]:
        out = []
        for task_id, by_arm in arm_results.items():
            r = by_arm.get(arm) or {}
            value = _holdout_delta(r)
            if value is None:
                continue
            out.append(PairedTaskResult(
                task_id=task_id,
                family_id=families.get(task_id, "unknown"),
                harness_sha=str((r.get("provenance") or {}).get("harness_sha256", "")),
                arm=arm,
                metrics={metrics: value},
                ok=bool(r.get("ok")),
                failure_kind=("infrastructure" if not r.get("ok") else ""),
                artifact_sha256=str(r.get("artifact_sha256", "")),
            ))
        return out

    n_pairs = sum(1 for by_arm in arm_results.values()
                  if _holdout_delta(by_arm.get("candidate") or {}) is not None)
    if n_pairs < 2:
        # Paired t / bootstrap / permutation inference is undefined with fewer
        # than two pairs. Do NOT fabricate statistics or silently pass: the
        # verdict step will record INCONCLUSIVE for insufficient evidence.
        return {
            "evaluator_version": EVALUATOR_VERSION,
            "preregistration_sha256": prereg.to_dict()["preregistration_sha256"],
            "primary_metric": metrics,
            "n_tasks": len(task_evaluations),
            "n_pairs": n_pairs,
            "insufficient_for_paired_inference": True,
            "comparisons": {},
            "statistical_analysis_sha256": sha256_json(
                {"n_pairs": n_pairs, "insufficient": True}),
        }

    comparisons = {
        "candidate_vs_base": analyse_paired(
            prereg=pre, base=_results("base"), cand=_results("candidate"),
            families=families),
        "candidate_vs_placebo": analyse_paired(
            prereg=pre, base=_results("placebo"), cand=_results("candidate"),
            families=families),
        "candidate_vs_shuffled": analyse_paired(
            prereg=pre, base=_results("shuffled"), cand=_results("candidate"),
            families=families),
    }
    return {
        "evaluator_version": EVALUATOR_VERSION,
        "preregistration_sha256": prereg.to_dict()["preregistration_sha256"],
        "primary_metric": metrics,
        "n_tasks": len(task_evaluations),
        "comparisons": comparisons,
        "statistical_analysis_sha256": sha256_json(comparisons),
    }


def decide_verdict(*, task_evaluations: list[dict[str, Any]],
                   statistics: dict[str, Any],
                   prereg: Preregistration) -> dict[str, Any]:
    """Evidence-bounded verdict. Missing evidence is NEVER converted to PASS."""
    reasons: list[str] = []
    n = len(task_evaluations)
    if n < prereg.min_n_tasks:
        reasons.append(f"n_tasks={n} < preregistered {prereg.min_n_tasks}")

    blocked = []
    for e in task_evaluations:
        v = e.get("experimental_validity") or {}
        if not v.get("all_arms_present") or not v.get("all_arms_ok"):
            blocked.append(e["task_id"])
        if v.get("ground_truth_leak"):
            reasons.append(f"ground truth leak: {e['task_id']}")
        if v.get("infrastructure_failures"):
            blocked.append(f"{e['task_id']}:infra")
    if blocked:
        reasons.append(f"blocked tasks/arms: {blocked[:5]}")

    def _mean(dim: str) -> float | None:
        vals = [_num(e["dimensions"].get(dim)) for e in task_evaluations]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    means = {dim: _mean(dim) for dim in DIMENSIONS}
    correctness = means.get("scientific_correctness")
    evidence = means.get("evidence_completeness")
    validity = means.get("experimental_validity")
    repro = means.get("reproducibility")

    cmp = statistics.get("comparisons") or {}
    primary = cmp.get("candidate_vs_base", {}).get("contrasts", [{}])[0] or {}
    p_base = _num(primary.get("holm_corrected_p"))
    significant = bool(primary.get("significant_after_correction"))
    delta_base = _num(primary.get("mean_difference"))
    placebo = cmp.get("candidate_vs_placebo", {}).get("contrasts", [{}])[0] or {}
    shuffled = cmp.get("candidate_vs_shuffled", {}).get("contrasts", [{}])[0] or {}
    delta_placebo = _num(placebo.get("mean_difference"))
    delta_shuffled = _num(shuffled.get("mean_difference"))

    verdict = "INCONCLUSIVE"
    if blocked or correctness is None:
        verdict = "BLOCKED"
    elif (correctness < 0.5 or (delta_base is not None and delta_base < 0)
          or (delta_placebo is not None and delta_placebo < 0)):
        verdict = "FAIL"
        reasons.append("candidate not supported against correctness/controls")
    elif (n >= prereg.min_n_tasks and significant and p_base is not None
          and p_base <= prereg.alpha
          and correctness is not None and correctness >= prereg.min_correctness
          and evidence is not None and evidence >= prereg.min_dimension_score
          and validity is not None and validity >= prereg.min_dimension_score
          and repro is not None and repro >= prereg.min_dimension_score
          and (delta_placebo is None or delta_placebo > 0)
          and (delta_shuffled is None or delta_shuffled > 0)):
        verdict = "PASS"
    else:
        reasons.append("pre-registered evidence gates not all satisfied")

    return {
        "verdict": verdict,
        "reasons": reasons,
        "dimension_means": {k: (round(v, 6) if v is not None else None)
                            for k, v in means.items()},
        "gates": {
            "primary_significant": significant,
            "primary_holm_p": p_base,
            "primary_mean_difference": delta_base,
            "candidate_vs_placebo_delta": delta_placebo,
            "candidate_vs_shuffled_delta": delta_shuffled,
            "minimum_meaningful_effect": prereg.minimum_meaningful_effect,
            "min_n_tasks": prereg.min_n_tasks,
        },
        "verdict_sha256": sha256_json({
            "verdict": verdict, "reasons": reasons, "means": means}),
        "note": (
            "This verdict reports engineering evidence from the executed "
            "pilot only. It is NOT a scientific validation of the method."
        ),
    }


__all__ = [
    "EVALUATOR_VERSION", "DIMENSIONS", "VERDICTS", "Preregistration",
    "evaluate_task", "analyse_statistics", "decide_verdict",
]
