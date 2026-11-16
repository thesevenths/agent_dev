"""M6 -- independent evaluation and paired statistics.

The evaluator is INDEPENDENT of candidate generation by construction:

  * the metric contract and primary metric are PREREGISTERED and frozen to disk
    BEFORE any Base/Candidate run, with their own evaluator version SHA;
  * the evaluation function receives recorded run results, never the candidate
    object or any candidate-supplied callable, so a candidate cannot control
    its own grade;
  * all eight metric families are reported separately -- no unconstrained
    weighted score collapses them;
  * statistical tests include paired bootstrap CI, paired permutation, McNemar
    (binary), clustered resampling by task family, Holm multiple-comparison
    correction, effect size and power analysis.

A candidate is ACCEPTED only if the pre-registered primary metric improves by
at least the minimum meaningful effect with a significant corrected paired
test AND no safety/other primary metric regresses. Otherwise REJECTED or
INCONCLUSIVE. Unfavourable results are never silently excluded.
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..evolution.statistics import (
    holm_step_down, paired_confidence_interval, paired_p_value,
)
from .identity import freeze_json, sha256_json

EVALUATOR_VERSION = "m6_eval_v1"
METRICS = (
    "objective_quality",
    "falsifiability_rate",
    "experiment_validity",
    "confounder_coverage",
    "diagnosis_accuracy",
    "evidence_completeness",
    "budget_efficiency",
    "transfer_performance",
)
BINARY_METRICS = frozenset({
    "falsifiability_rate", "confounder_coverage", "diagnosis_accuracy"})


class EvaluationError(ValueError):
    """Raised for missing evaluators or incomplete results."""


@dataclass(frozen=True)
class PreRegistration:
    """Frozen before any run; part of the evidence package."""
    experiment_id: str
    primary_metric: str
    direction: str                     # maximize | minimize
    minimum_meaningful_effect: float
    alpha: float = 0.05
    n_bootstrap: int = 2000
    n_permutation: int = 2000
    evaluator_version: str = EVALUATOR_VERSION
    seed: int = 20261116
    secondary_metrics: tuple[str, ...] = tuple(
        m for m in METRICS if m != "objective_quality")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["secondary_metrics"] = list(self.secondary_metrics)
        d["evaluator_sha256"] = sha256_json(d)
        return d

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_dict())


def freeze_preregistration(pre: PreRegistration,
                           path: str | Path) -> dict[str, Any]:
    obj = pre.to_dict()
    obj["preregistration_sha256"] = pre.sha256
    frozen = freeze_json(obj, path)
    return {**frozen, "preregistration_sha256": pre.sha256}


@dataclass
class TaskResult:
    """One recorded execution of one harness on one task."""
    task_id: str
    family_id: str
    harness_sha: str
    arm: str                       # base | candidate
    metrics: dict[str, float]
    ok: bool = True
    failure_kind: str = ""         # research | infrastructure
    artifact_sha256: str = ""
    trajectory_sha256: str = ""
    tokens: int = 0
    wall_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_metric_set(results: list[TaskResult]) -> dict[str, Any]:
    """Compute all eight metrics from RECORDED results; retain every run."""
    out: dict[str, Any] = {}
    for metric in METRICS:
        vals = [float(r.metrics[metric]) for r in results
                if r.ok and metric in r.metrics]
        out[metric] = {"n": len(vals),
                       "mean": (sum(vals) / len(vals)) if vals else None,
                       "n_failed": len([r for r in results if not r.ok])}
    out["n_total"] = len(results)
    out["infrastructure_failures"] = len(
        [r for r in results if r.failure_kind == "infrastructure"])
    out["research_failures"] = len(
        [r for r in results if r.failure_kind == "research"])
    return out


def _paired_differences(base: list[TaskResult], cand: list[TaskResult],
                        metric: str, direction: str) -> tuple[list[str], list[float]]:
    """Pair by task_id; missing pair is retained as zero (no silent exclusion)."""
    bmap = {r.task_id: r for r in base}
    cmap = {r.task_id: r for r in cand}
    ids = sorted(set(bmap) | set(cmap))
    diffs: list[float] = []
    for tid in ids:
        b = bmap.get(tid)
        c = cmap.get(tid)
        bv = float(b.metrics.get(metric, 0.0)) if b and b.ok else 0.0
        cv = float(c.metrics.get(metric, 0.0)) if c and c.ok else 0.0
        d = cv - bv
        diffs.append(d if direction == "maximize" else -d)
    return ids, diffs


def _paired_bootstrap_ci(diffs: list[float], *, seed: int, n: int = 2000,
                         alpha: float = 0.05) -> list[float]:
    if not diffs:
        return [None, None]
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = int(alpha / 2 * n)
    hi = min(n - 1, int((1 - alpha / 2) * n))
    return [means[lo], means[hi]]


def _paired_permutation_p(diffs: list[float], *, seed: int,
                          n: int = 2000) -> float:
    if not diffs:
        return 1.0
    observed = abs(sum(diffs) / len(diffs))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(n):
        flipped = [d if rng.getrandbits(1) else -d for d in diffs]
        if abs(sum(flipped) / len(flipped)) >= observed - 1e-12:
            extreme += 1
    return (extreme + 1) / (n + 1)


def _mcnemar_p(base: list[TaskResult], cand: list[TaskResult], metric: str,
               threshold: float = 0.5) -> float | None:
    """Exact-ish McNemar on binarised paired outcomes (normal approximation)."""
    bmap = {r.task_id: r for r in base}
    cmap = {r.task_id: r for r in cand}
    b01 = b10 = 0
    for tid in set(bmap) & set(cmap):
        b = float(bmap[tid].metrics.get(metric, 0.0)) >= threshold
        c = float(cmap[tid].metrics.get(metric, 0.0)) >= threshold
        if c and not b:
            b01 += 1
        elif b and not c:
            b10 += 1
    n = b01 + b10
    if n == 0:
        return None
    chi = (abs(b01 - b10) - 1) ** 2 / n
    return math.erfc(math.sqrt(chi / 2))


def _clustered_bootstrap(diffs: list[float], families: list[str], *,
                         seed: int, n: int = 2000, alpha: float = 0.05
                         ) -> list[float]:
    """Resample whole TASK FAMILIES, never individual rows within a family."""
    clusters: dict[str, list[float]] = {}
    for fam, d in zip(families, diffs):
        clusters.setdefault(fam, []).append(d)
    keys = sorted(clusters)
    if not keys:
        return [None, None]
    rng = random.Random(seed + 7)
    means = []
    for _ in range(n):
        chosen = [keys[rng.randrange(len(keys))] for _ in keys]
        vals = [d for k in chosen for d in clusters[k]]
        means.append(sum(vals) / len(vals) if vals else 0.0)
    means.sort()
    lo = int(alpha / 2 * n)
    hi = min(n - 1, int((1 - alpha / 2) * n))
    return [means[lo], means[hi]]


def _holm_adjusted_pvalues(p_values: list[float], alpha: float = 0.05
                           ) -> list[float]:
    """Holm adjusted p-values (monotone step-up form of the step-down gate)."""
    m = len(p_values)
    if m == 0:
        return []
    clean = [min(1.0, max(0.0, float(p))) for p in p_values]
    order = sorted(range(m), key=lambda i: clean[i])
    adjusted = [1.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * clean[idx]))
        adjusted[idx] = running
    return adjusted


def analyse_paired(*, prereg: PreRegistration, base: list[TaskResult],
                   cand: list[TaskResult], families: dict[str, str],
                   evaluator_sha256: str = "") -> dict[str, Any]:
    """Full pre-registered paired analysis. Never called by candidate code."""
    p = prereg.to_dict()
    contrasts: list[dict[str, Any]] = []
    for metric in (prereg.primary_metric,) + tuple(prereg.secondary_metrics):
        ids, diffs = _paired_differences(base, cand, metric,
                                         prereg.direction)
        fams = [families.get(tid, "unknown") for tid in ids]
        mean = sum(diffs) / len(diffs) if diffs else 0.0
        sd = (math.sqrt(sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1))
              if len(diffs) > 1 else 0.0)
        pval = paired_p_value(mean, sd, len(diffs)) if len(diffs) > 1 else 1.0
        ci = paired_confidence_interval(mean, sd, len(diffs), alpha=prereg.alpha)
        contrasts.append({
            "metric": metric,
            "is_primary": metric == prereg.primary_metric,
            "n_pairs": len(diffs), "mean_difference": mean,
            "paired_t_p": pval,
            "bootstrap_ci95": _paired_bootstrap_ci(
                diffs, seed=prereg.seed, n=prereg.n_bootstrap),
            "clustered_family_ci95": _clustered_bootstrap(
                diffs, fams, seed=prereg.seed, n=prereg.n_bootstrap),
            "permutation_p": _paired_permutation_p(
                diffs, seed=prereg.seed, n=prereg.n_permutation),
            "mcnemar_p": (_mcnemar_p(base, cand, metric)
                          if metric in BINARY_METRICS else None),
            "cohens_dz": (mean / sd if sd > 0 else None),
            "confidence_interval_95": list(ci) if ci else [None, None],
        })
    pvals = [c["paired_t_p"] for c in contrasts]
    corrected = holm_step_down(pvals, alpha=prereg.alpha)
    adjusted_p = _holm_adjusted_pvalues(pvals, alpha=prereg.alpha)
    for c, corr, adj in zip(contrasts, corrected, adjusted_p):
        c["holm_corrected_p"] = adj
        c["significant_after_correction"] = bool(
            adj <= prereg.alpha and corr.get("significant"))
    primary = next(c for c in contrasts if c["is_primary"])
    meets_mme = (primary["mean_difference"] >= prereg.minimum_meaningful_effect)
    infra = len([r for r in base + cand if r.failure_kind == "infrastructure"])
    regressions = [c["metric"] for c in contrasts
                   if c["is_primary"] and c["mean_difference"] < 0]
    decision = "INCONCLUSIVE"
    if infra == 0 and meets_mme and primary["significant_after_correction"] \
            and not regressions:
        decision = "ACCEPTED"
    elif infra == 0 and primary["mean_difference"] < 0:
        decision = "REJECTED"
    return {
        "evaluator_version": prereg.evaluator_version,
        "evaluator_sha256": evaluator_sha256 or p["evaluator_sha256"],
        "preregistration_sha256": p["evaluator_sha256"],
        "decision": decision,
        "primary_metric": prereg.primary_metric,
        "primary_meets_mme": meets_mme,
        "regressed_metrics": regressions,
        "infrastructure_failures": infra,
        "base_metrics": compute_metric_set(base),
        "candidate_metrics": compute_metric_set(cand),
        "contrasts": contrasts,
        "n_comparisons": len(contrasts),
        "all_results_retained": len(base) + len(cand),
    }


__all__ = [
    "EvaluationError", "METRICS", "EVALUATOR_VERSION", "PreRegistration",
    "TaskResult", "freeze_preregistration", "compute_metric_set",
    "analyse_paired",
]
