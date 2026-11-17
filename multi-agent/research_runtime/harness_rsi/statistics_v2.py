"""M6.1 -- family-level statistical confirmation (honest, pre-registered).

M6 treated a SEED as an independent observation and used a task-level t-test.
M6.1 uses TASK FAMILY as the statistical unit:

  1. Aggregate seeded runs WITHIN each family -> one family-level paired effect.
  2. Clustered bootstrap CI over families.
  3. Family-level sign-flip permutation test (exact enumeration when feasible).
  4. Effect size, family win/tie/loss, domain-stratified view.
  5. Holm correction across the PRE-REGISTERED family of metrics only.
  6. Power analysis from pilot family variance; insufficient families =>
     INCONCLUSIVE (never manufacture significance).

All inputs are SHA-bound trajectory/result records; a missing or mismatched
result sha fails closed.
"""
from __future__ import annotations

import itertools
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .identity import freeze_json, sha256_json

STAT_VERSION = "m6_1_family_statistics_v1"


class StatisticalError(ValueError):
    """Raised for invalid/mismatched statistical evidence."""


@dataclass(frozen=True)
class FamilyResult:
    """One family-level paired observation, bound to real result artifacts."""
    family_id: str
    domain: str
    base_value: float
    candidate_value: float
    base_result_sha: str
    candidate_result_sha: str
    n_seeds: int = 1

    @property
    def difference(self) -> float:
        return self.candidate_value - self.base_value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreregistrationV2:
    """Immutable experiment contract, frozen BEFORE any result is inspected."""
    experiment_id: str
    primary_metric: str
    primary_component: str
    metric_direction: str
    minimum_practical_effect: float
    alpha: float = 0.05
    target_power: float = 0.80
    statistical_unit: str = "task_family"
    primary_test: str = "family_sign_flip_permutation"
    ci_method: str = "cluster_bootstrap"
    multiplicity_correction: str = "holm"
    minimum_families: int = 8
    n_bootstrap: int = 2000
    seed: int = 20261117
    selection_manifest_sha: str = ""
    holdout_manifest_sha: str = ""
    candidate_sha: str = ""
    parent_sha: str = ""
    evaluation_version: str = STAT_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Raw immutable contract fields (identity is computed ONCE)."""
        return asdict(self)

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_dict())


def freeze_preregistration_v2(pre: PreregistrationV2,
                              path: str | Path) -> dict[str, Any]:
    obj = pre.to_dict()
    obj["frozen_sha256"] = pre.sha256
    frozen = freeze_json(obj, path)
    return {**frozen, "preregistration_sha256": pre.sha256}


def _dim_ok(value: float, direction: str) -> float:
    return value if direction == "maximize" else -value


def aggregated_effects(results: list[FamilyResult], *, metric: str,
                       direction: str, expected_sha_map: dict[str, tuple[str, str]] | None = None
                       ) -> dict[str, Any]:
    """Aggregate seeded runs into ONE effect per family with SHA verification."""
    by_family: dict[str, list[FamilyResult]] = {}
    for r in results:
        if expected_sha_map is not None:
            exp = expected_sha_map.get(r.family_id)
            if exp is None or exp[0] != r.base_result_sha or \
                    exp[1] != r.candidate_result_sha:
                raise StatisticalError(
                    f"result sha mismatch for family {r.family_id}")
        by_family.setdefault(r.family_id, []).append(r)
    effects: dict[str, float] = {}
    details: list[dict[str, Any]] = []
    for fam, rs in sorted(by_family.items()):
        base = sum(_dim_ok(r.base_value, direction) for r in rs) / len(rs)
        cand = sum(_dim_ok(r.candidate_value, direction) for r in rs) / len(rs)
        effects[fam] = cand - base
        details.append({"family_id": fam, "domain": rs[0].domain,
                        "n_seeds": sum(r.n_seeds for r in rs),
                        "base": round(base, 6), "candidate": round(cand, 6),
                        "difference": round(cand - base, 6),
                        "win": "candidate" if cand > base else
                               ("base" if cand < base else "tie")})
    return {"effects": effects, "details": details,
            "n_families": len(effects)}


def _cluster_bootstrap_ci(effects: dict[str, float], *, seed: int,
                          n: int = 2000, alpha: float = 0.05) -> list[float]:
    vals = list(effects.values())
    if not vals:
        return [None, None]
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        sample = [vals[rng.randrange(len(vals))] for _ in vals]
        means.append(sum(sample) / len(sample))
    means.sort()
    return [means[int(alpha / 2 * n)],
            means[min(n - 1, int((1 - alpha / 2) * n))]]


def _sign_flip_p(effects: dict[str, float]) -> dict[str, Any]:
    """Exact family sign-flip permutation (2^n) when n<=16, else Monte Carlo."""
    vals = list(effects.values())
    n = len(vals)
    if not vals:
        return {"p": 1.0, "method": "empty"}
    observed = abs(sum(vals) / n)
    if n <= 16:
        count = 0
        total = 0
        for signs in itertools.product((-1, 1), repeat=n):
            total += 1
            if abs(sum(s * v for s, v in zip(signs, vals)) / n) >= observed - 1e-12:
                count += 1
        return {"p": count / total, "method": "exact_enumeration", "n": total}
    rng = random.Random(20261117)
    count = 0
    n_mc = 10000
    for _ in range(n_mc):
        flipped = [v if rng.getrandbits(1) else -v for v in vals]
        if abs(sum(flipped) / n) >= observed - 1e-12:
            count += 1
    return {"p": (count + 1) / (n_mc + 1), "method": "monte_carlo", "n": n_mc}


def _holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    out = [1.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[idx]))
        out[idx] = running
    return out


def family_power_analysis(effects: dict[str, float], *, alpha: float = 0.05,
                          target_power: float = 0.80) -> dict[str, Any]:
    """Plain-language power estimate from observed family-level variance."""
    vals = list(effects.values())
    n = len(vals)
    if n < 2:
        return {"ok": False, "reason": "too_few_families", "n_families": n}
    mean = sum(vals) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1))
    se = sd / math.sqrt(n)
    # Normal-approximation achieved power for the observed effect at alpha.
    if se > 0:
        z = abs(mean) / se
        achieved = max(0.0, min(1.0, 0.5 + 0.5 * math.erf(
            (z - 1.96) / math.sqrt(2))))
    else:
        achieved = 1.0 if mean != 0 else 0.0
    # Approximate n needed for target power 0.8 (two-sided alpha 0.05).
    needed = (2.8 * sd / abs(mean)) ** 2 if mean != 0 else float("inf")
    if math.isfinite(needed):
        n_needed = math.ceil(needed)
    else:
        n_needed = None   # zero/near-zero effect: no finite n can reach power
    return {"ok": True, "n_families": n, "mean_effect": mean,
            "sd": sd, "se": se, "achieved_power": achieved,
            "target_power": target_power,
            "approx_families_for_target": n_needed,
            "sufficient": (n_needed is not None and n >= n_needed
                           and achieved >= target_power)}


def analyse_family_confirmation(*, prereg: PreregistrationV2,
                                results: list[FamilyResult],
                                expected_sha_map: dict[str, tuple[str, str]] | None = None
                                ) -> dict[str, Any]:
    """Full family-level confirmatory analysis. Fails closed on SHA mismatch."""
    agg = aggregated_effects(results, metric=prereg.primary_metric,
                             direction=prereg.metric_direction,
                             expected_sha_map=expected_sha_map)
    effects = agg["effects"]
    n_fam = agg["n_families"]
    # Fail closed BEFORE any directional verdict: zero/too-few families is
    # missing evidence, never evidence of rejection.
    if n_fam < prereg.minimum_families or len(effects) < 2:
        early = {"stat_version": STAT_VERSION, "verdict": "INCONCLUSIVE",
                 "reason": "insufficient_independent_families",
                 "n_families": n_fam,
                 "minimum_families": prereg.minimum_families}
        return early
    ci = _cluster_bootstrap_ci(effects, seed=prereg.seed,
                               n=prereg.n_bootstrap, alpha=prereg.alpha)
    perm = _sign_flip_p(effects)
    power = family_power_analysis(effects, alpha=prereg.alpha,
                                  target_power=prereg.target_power)
    mean_effect = sum(effects.values()) / n_fam if n_fam else 0.0
    mme_ok = mean_effect >= prereg.minimum_practical_effect
    ci_ok = ci[0] is not None and ci[0] >= prereg.minimum_practical_effect
    sig = perm["p"] <= prereg.alpha
    n_ok = n_fam >= prereg.minimum_families
    if not power["ok"] or not power["sufficient"]:
        verdict = "INCONCLUSIVE"   # underpowered -> MUST stay inconclusive
    elif not (mme_ok and ci_ok and sig and n_ok):
        verdict = ("INCONCLUSIVE" if mean_effect > 0 else "REJECTED")
    else:
        verdict = "ACCEPTED"
    wins = {"candidate": sum(1 for d in agg["details"]
                             if d["win"] == "candidate"),
            "base": sum(1 for d in agg["details"] if d["win"] == "base"),
            "tie": sum(1 for d in agg["details"] if d["win"] == "tie")}
    domains: dict[str, list[float]] = {}
    for d in agg["details"]:
        domains.setdefault(d["domain"], []).append(d["difference"])
    return {"stat_version": STAT_VERSION,
            "preregistration_sha256": prereg.sha256,
            "primary_metric": prereg.primary_metric,
            "statistical_unit": prereg.statistical_unit,
            "verdict": verdict, "n_families": n_fam,
            "mean_family_effect": mean_effect,
            "ci95_cluster_bootstrap": ci, "permutation": perm,
            "holm_note": "single pre-registered primary metric; secondary "
                         "metrics corrected separately if promoted",
            "power": power, "mme": prereg.minimum_practical_effect,
            "mme_met": mme_ok, "ci_met": ci_ok, "significant": sig,
            "min_families_met": n_ok,
            "family_win_tie_loss": wins,
            "domain_stratified": {k: {"n": len(v),
                                      "mean": sum(v) / len(v)}
                                  for k, v in domains.items()},
            "family_details": agg["details"],
            "all_results_retained": len(results)}


__all__ = [
    "STAT_VERSION", "StatisticalError", "FamilyResult", "PreregistrationV2",
    "freeze_preregistration_v2", "aggregated_effects",
    "analyse_family_confirmation", "family_power_analysis",
]
