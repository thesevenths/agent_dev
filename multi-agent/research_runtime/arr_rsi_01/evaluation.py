"""ARR-RSI-01 P0.1 -- independent evaluation with INTEGRITY gates.

Only this module reads sealed ground truth. It validates evidence references
against real files, records per-arm infrastructure failures, keeps missing data
visible, treats seed replicates as non-independent, and refuses to convert
BLOCKED into PASS.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..harness_rsi.evaluation import (
    PreRegistration, TaskResult as PairedTaskResult, analyse_paired,
)
from .arms import ARMS, ArmSpec, assert_matched_budgets
from .diagnosis import REQUIRED_DIAGNOSIS_FIELDS
from .tasks import (
    GroundTruth, INDEPENDENCE_UNIT, PublicTask, assert_no_ground_truth,
    sha256_json,
)

EVALUATOR_VERSION = "arr_rsi_01_p0_1_eval_v1"
DIMENSIONS = (
    "scientific_correctness", "ood_performance", "evidence_completeness",
    "experimental_validity", "reproducibility", "budget_efficiency",
)
VERDICTS = ("PASS", "FAIL", "INCONCLUSIVE", "BLOCKED")

# Evaluator-side ONLY: map evidence-derived cause vocabulary onto the sealed
# author-side root-cause taxonomy. Never imported by Research/Diagnosis.
CAUSE_TO_SEALED: dict[str, str] = {
    "dynamics_error_accumulation": "latent_dynamics_undertrained",
    "representation_capacity_limited": "insufficient_representation_capacity",
    "ood_generalization_gap": "insufficient_representation_capacity",
    "weak_prediction_signal": "weak_prediction_objective",
}


@dataclass(frozen=True)
class Preregistration:
    evaluator_version: str = EVALUATOR_VERSION
    primary_metric: str = "arr_rsi_holdout_delta"
    direction: str = "maximize"
    minimum_meaningful_effect: float = 0.02
    alpha: float = 0.05
    min_n_tasks: int = 5
    min_n_independent_units: int = 5
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
    return v if math.isfinite(v) else None


def _holdout_delta(arm_result: dict[str, Any]) -> float | None:
    evidence = arm_result.get("evidence") or {}
    for key in ("holdout", "ood"):
        value = _num((evidence.get(key) or {}).get("mean_holdout_delta"))
        if value is not None:
            return value
    return _num(((arm_result.get("telemetry") or {}).get("ood") or {}).get(
        "mean_delta"))


def _resolve_refs(refs: list[str]) -> dict[str, Any]:
    resolved, unresolved = [], []
    for ref in refs:
        suffix = str(ref).split(":", 1)[-1]
        candidate = Path(suffix) if suffix and ("/" in suffix or "\\" in suffix) \
            else None
        known = any(tok in str(ref) for tok in (
            "baseline_id:", "tool_call:", "shuffled_from:",
            "experiment_id:", "job_id:", "n_evidence_records:"))
        (resolved if (candidate and candidate.exists()) or known
         else unresolved).append(str(ref))
    return {"n_refs": len(refs), "n_resolved": len(resolved),
            "n_unresolved": len(unresolved), "unresolved": unresolved[:10],
            "score": (len(resolved) / len(refs)) if refs else 0.0}


def _cause_score(truth: GroundTruth, diagnosis: dict[str, Any]) -> dict[str, Any]:
    evidence_cause = str(diagnosis.get("root_cause") or "")
    mapped = CAUSE_TO_SEALED.get(evidence_cause, evidence_cause)
    cause_ok = mapped == truth.ground_truth_root_cause
    iv = diagnosis.get("proposed_intervention") or {}
    return {"root_cause_correct": bool(cause_ok),
            "evidence_cause": evidence_cause, "mapped_cause": mapped,
            "diagnosis_key": iv.get("key") if isinstance(iv, dict) else None,
            "truth_key": truth.correct_intervention,
            "score": 1.0 if cause_ok else 0.0}


def evaluate_task(*, task: PublicTask, truth: GroundTruth,
                  arms: dict[str, ArmSpec],
                  arm_results: dict[str, dict[str, Any]],
                  diagnoses: dict[str, dict[str, Any]],
                  trajectories: dict[str, dict[str, Any]],
                  prereg: Preregistration) -> dict[str, Any]:
    """Evaluate ONE task. GT is used here and nowhere else."""
    for payload in (arm_results, diagnoses, trajectories,
                    [a.to_dict() for a in arms.values()]):
        assert_no_ground_truth(payload)

    base_traj = trajectories.get("base") or {}
    d = (diagnoses.get(task.task_id) or {}).get("base") or {}
    scientific = _cause_score(truth, d)
    evidence = _resolve_refs(list(d.get("evidence_refs") or []))

    matched = {"matched": False, "signature": {}}
    try:
        matched = assert_matched_budgets(
            arms, seeds=list(task.budget.get("paired_seeds", [task.seed])),
            timeout_s=float(task.budget.get("max_total_seconds", 900.0)))
    except ValueError:
        pass

    infra = {name: list((traj.get("infrastructure_failures") or []))
             for name, traj in trajectories.items()}
    infra_count = sum(len(v) for v in infra.values())
    checks = []
    for name, arm in arms.items():
        record = arm.intervention_record or {}
        checks.append({"arm": name, "status": record.get("status", "MISSING"),
                       "application_verified": bool(record.get(
                           "application_verified")),
                       "ok": bool((arm_results.get(name) or {}).get("ok")),
                       "missing": name not in arm_results})
    candidate_valid = next((v for v in checks if v["arm"] == "candidate"), {})
    sd = (diagnoses.get(task.task_id) or {}).get("shuffled") or {}
    shuffled_valid = bool(sd.get("shuffled_from")) and (
        (sd.get("proposed_intervention") or {}).get("key")
        != (d.get("proposed_intervention") or {}).get("key"))
    all_ok = all(bool((arm_results.get(n) or {}).get("ok")) for n in ARMS)
    all_present = all(n in arm_results for n in ARMS)
    valid = bool(matched["matched"] and all_present and all_ok and not infra_count
                 and shuffled_valid
                 and candidate_valid.get("status") in ("APPLIED", "INERT"))
    if candidate_valid.get("status") == "APPLIED" and not candidate_valid.get(
            "application_verified"):
        valid = False

    reproducibility = {
        "task_sha256": bool(task.independence_unit),
        "frozen_dataset_version": bool(task.dataset_version),
        "seeds_recorded": bool(task.budget.get("paired_seeds")),
        "run_scoped_ids": bool(base_traj.get("run_id")),
        "artifact_sha256": bool(all((arm_results.get(n) or {}).get(
            "artifact_sha256") for n in ARMS)),
    }
    repro_score = sum(bool(v) for v in reproducibility.values()) / len(reproducibility)
    candidate = arm_results.get("candidate") or {}
    jobs = sum(int((r.get("telemetry") or {}).get("jobs_used", 0) or 0)
               for r in arm_results.values())
    delta = _holdout_delta(candidate)
    quality = _num((candidate.get("quality") or {}).get("Q"))
    efficiency = round(quality / jobs, 6) if quality is not None and jobs > 0 else None
    complete_case = bool(all_present and all_ok and not infra_count
                         and all(v["status"] != "NOT_APPLIED" for v in checks)
                         and not evidence["n_unresolved"])
    dims = {"scientific_correctness": scientific["score"],
            "ood_performance": 1.0 if (delta is not None and delta > 0) else 0.0,
            "evidence_completeness": round(evidence["score"], 4),
            "experimental_validity": 1.0 if valid else 0.0,
            "reproducibility": round(repro_score, 4),
            "budget_efficiency": efficiency}
    return {
        "task_id": task.task_id, "task_family": task.task_family,
        "independence_unit": task.independence_unit, "dimensions": dims,
        "scientific_correctness": scientific, "evidence": evidence,
        "intervention_validity": {
            "valid": bool(candidate_valid.get("status") in ("APPLIED", "INERT")
                          and valid),
            "applied_status": candidate_valid.get("status", "MISSING"),
            "checks": checks, "shuffled_distinct": shuffled_valid},
        "infrastructure": {"per_arm": infra, "n_failures": infra_count},
        "experimental_validity": {"valid": valid, "matched": matched,
                                  "all_arms_present": all_present,
                                  "all_arms_ok": all_ok,
                                  "shuffled_valid": shuffled_valid},
        "reproducibility_detail": reproducibility, "complete_case": complete_case,
        "missing_data_retained": not complete_case,
        "budget": {"jobs": jobs, "quality": quality, "efficiency": efficiency},
        "ground_truth_used_only_here": True}


def analyse_statistics(*, task_evaluations: list[dict[str, Any]],
                       arm_results: dict[str, dict[str, dict[str, Any]]],
                       prereg: Preregistration) -> dict[str, Any]:
    pre = PreRegistration(
        experiment_id="arr_rsi_01_p0.1", primary_metric=prereg.primary_metric,
        direction=prereg.direction,
        minimum_meaningful_effect=prereg.minimum_meaningful_effect,
        alpha=prereg.alpha, secondary_metrics=())
    independence = {e["task_id"]: e.get("independence_unit", "unknown")
                    for e in task_evaluations}
    units = len(set(independence.values()))
    n_pairs = sum(1 for by_arm in arm_results.values()
                  if _holdout_delta(by_arm.get("candidate") or {}) is not None)
    base_block = {"evaluator_version": EVALUATOR_VERSION,
                  "preregistration_sha256": prereg.to_dict()[
                      "preregistration_sha256"],
                  "primary_metric": prereg.primary_metric,
                  "n_tasks": len(task_evaluations), "n_pairs": n_pairs,
                  "n_independent_units": units,
                  "n_seed_replicates_per_unit": len(task_evaluations) // max(1, units),
                  "independence_unit": INDEPENDENCE_UNIT}
    if n_pairs < 2:
        return {**base_block, "insufficient_for_paired_inference": True,
                "comparisons": {},
                "statistical_analysis_sha256": sha256_json(
                    {"n_pairs": n_pairs, "insufficient": True})}

    def _results(arm: str) -> list[PairedTaskResult]:
        out = []
        for task_id, by_arm in arm_results.items():
            r = by_arm.get(arm) or {}
            value = _holdout_delta(r)
            if value is None:
                continue
            out.append(PairedTaskResult(
                task_id=task_id, family_id=independence.get(task_id, "unknown"),
                harness_sha=str((r.get("provenance") or {}).get(
                    "harness_sha256", "")), arm=arm,
                metrics={prereg.primary_metric: value}, ok=bool(r.get("ok")),
                failure_kind="infrastructure" if not r.get("ok") else "",
                artifact_sha256=str(r.get("artifact_sha256", ""))))
        return out

    comparisons = {
        f"candidate_vs_{name}": analyse_paired(
            prereg=pre, base=_results(name), cand=_results("candidate"),
            families=independence)
        for name in ("base", "placebo", "shuffled")}
    return {**base_block,
            "note": ("Seed replicates clustered by family|dataset; inference "
                     "unit is the independent research unit."),
            "comparisons": comparisons,
            "statistical_analysis_sha256": sha256_json(comparisons)}


def decide_verdict(*, task_evaluations: list[dict[str, Any]],
                   statistics: dict[str, Any],
                   prereg: Preregistration) -> dict[str, Any]:
    """BLOCKED first. Missing evidence can never become PASS."""
    reasons: list[str] = []
    n = len(task_evaluations)
    if n < prereg.min_n_tasks:
        reasons.append(f"n_tasks={n} < preregistered {prereg.min_n_tasks}")
    n_units = int(statistics.get("n_independent_units") or 0)
    if n_units < prereg.min_n_independent_units:
        reasons.append(f"n_independent_units={n_units} < "
                       f"{prereg.min_n_independent_units}")
    blocked = []
    for e in task_evaluations:
        v = e.get("experimental_validity") or {}
        infra = e.get("infrastructure") or {}
        iv = e.get("intervention_validity") or {}
        if not v.get("all_arms_present") or not v.get("all_arms_ok"):
            blocked.append(e["task_id"])
        if infra.get("n_failures"):
            blocked.append(f"{e['task_id']}:infra")
        if iv.get("applied_status") == "NOT_APPLIED":
            blocked.append(f"{e['task_id']}:not_applied")
        if not iv.get("shuffled_distinct"):
            reasons.append(f"shuffled diagnosis not distinct: {e['task_id']}")
    if blocked:
        reasons.append(f"blocked tasks/arms: {sorted(set(blocked))[:6]}")

    def _mean(dim: str) -> float | None:
        vals = [_num(e["dimensions"].get(dim)) for e in task_evaluations]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    means = {dim: _mean(dim) for dim in DIMENSIONS}
    correctness, evidence = means.get("scientific_correctness"), means.get(
        "evidence_completeness")
    validity, repro = means.get("experimental_validity"), means.get("reproducibility")
    complete = all(bool(e.get("complete_case")) for e in task_evaluations)
    comparisons = statistics.get("comparisons") or {}
    primary = comparisons.get("candidate_vs_base", {}).get(
        "contrasts", [{}])[0] or {}
    p_base = _num(primary.get("holm_corrected_p"))
    significant = bool(primary.get("significant_after_correction"))
    delta_base = _num(primary.get("mean_difference"))
    dp = _num((comparisons.get("candidate_vs_placebo", {}).get(
        "contrasts", [{}])[0] or {}).get("mean_difference"))
    ds = _num((comparisons.get("candidate_vs_shuffled", {}).get(
        "contrasts", [{}])[0] or {}).get("mean_difference"))

    verdict = "INCONCLUSIVE"
    if blocked or correctness is None or statistics.get(
            "insufficient_for_paired_inference"):
        verdict = "BLOCKED"
    elif (correctness < 0.5 or (delta_base is not None and delta_base < 0)
          or (dp is not None and dp < 0)):
        verdict = "FAIL"
        reasons.append("candidate not supported against correctness/controls")
    elif (n >= prereg.min_n_tasks and n_units >= prereg.min_n_independent_units
          and complete and significant and p_base is not None
          and p_base <= prereg.alpha
          and correctness >= prereg.min_correctness
          and evidence is not None and evidence >= prereg.min_dimension_score
          and validity is not None and validity >= prereg.min_dimension_score
          and repro is not None and repro >= prereg.min_dimension_score
          and delta_base is not None
          and delta_base >= prereg.minimum_meaningful_effect
          and (dp is None or dp > 0) and (ds is None or ds > 0)):
        verdict = "PASS"
    else:
        reasons.append("pre-registered integrity/statistical gates not all met")
    return {
        "verdict": verdict, "reasons": reasons,
        "dimension_means": {k: (round(v, 6) if v is not None else None)
                            for k, v in means.items()},
        "gates": {"primary_significant": significant, "primary_holm_p": p_base,
                  "primary_mean_difference": delta_base,
                  "candidate_vs_placebo_delta": dp,
                  "candidate_vs_shuffled_delta": ds,
                  "minimum_meaningful_effect": prereg.minimum_meaningful_effect,
                  "n_independent_units": n_units, "complete_case": complete,
                  "min_n_tasks": prereg.min_n_tasks},
        "verdict_sha256": sha256_json(
            {"verdict": verdict, "reasons": reasons, "means": means}),
        "note": ("Engineering evidence only; NOT scientific validation. "
                 "Original P0 BLOCKED verdict is preserved unchanged.")}


__all__ = ["EVALUATOR_VERSION", "DIMENSIONS", "VERDICTS", "Preregistration",
           "evaluate_task", "analyse_statistics", "decide_verdict"]
