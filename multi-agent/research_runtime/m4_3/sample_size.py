"""M4.3-02 -- sample-size estimation for the paired-block A/B/C design.

Design facts this module respects
---------------------------------
  * the independent unit is a BLOCK (one research opportunity with paired A/B/C
    arms), not an individual run;
  * the estimand is a paired difference of a frozen primary metric
    (B-A, C-B, C-A), so the variance that matters is the variance of the paired
    per-block differences;
  * three contrasts are tested, so alpha must be corrected (Holm in the
    confirmatory analysis; the POWER calculation uses the conservative
    Bonferroni-equivalent alpha to stay on the safe side);
  * the effect assumption is a PRE-SPECIFIED smallest effect of interest
    (SEOI). The largest effect observed in the pilot is explicitly NOT used --
    that would guarantee an underpowered confirmatory run;
  * variance is inflated conservatively and a sensitivity grid is reported,
    because a pilot variance is itself estimated from very few blocks;
  * if the pilot does not yield enough paired blocks to estimate a variance,
    the module says so and refuses to invent a number.

Nothing here runs an experiment or looks at a p-value. Stdlib only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, variance
from typing import Any, Iterable

# Two-sided normal quantiles (sufficient for a power calculation; the analysis
# itself uses a Student-t critical value).
_Z = {0.80: 0.8416, 0.85: 1.0364, 0.90: 1.2816, 0.95: 1.6449, 0.975: 1.9600,
      0.99: 2.3263, 0.995: 2.5758}


def z_quantile(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation)."""
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p must be in (0, 1)")
    if p in _Z:
        return _Z[p]
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
                + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
                 + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r +
            a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r +
                          b[4]) * r + 1)


#: Refuse to estimate a variance from fewer paired blocks than this.
MIN_BLOCKS_FOR_VARIANCE = 3


@dataclass
class ContrastSpec:
    """One pre-registered contrast and its smallest effect of interest."""
    name: str
    seoi: float
    metric: str = "independently_reviewed_objective_quality"


def blocks_required(*, variance_d: float, seoi: float, alpha: float,
                    power: float, inflation: float = 1.0) -> int:
    """Blocks needed for a two-sided paired test of one contrast."""
    if seoi <= 0:
        raise ValueError("the smallest effect of interest must be positive")
    if not (0 < alpha < 1) or not (0 < power < 1):
        raise ValueError("alpha and power must be in (0, 1)")
    z_alpha = z_quantile(1 - alpha / 2)
    z_beta = z_quantile(power)
    var_inflated = variance_d * inflation
    n = ((z_alpha + z_beta) ** 2) * var_inflated / (seoi ** 2)
    return max(1, int(math.ceil(n)))


def summarise(values: Iterable[float]) -> dict[str, Any]:
    vals = [float(v) for v in values]
    if not vals:
        return {"n": 0, "mean": None, "sd": None, "variance": None}
    return {"n": len(vals), "mean": round(mean(vals), 6),
            "sd": round(math.sqrt(variance(vals)), 6) if len(vals) > 1 else 0.0,
            "variance": round(variance(vals), 6) if len(vals) > 1 else 0.0}


def estimate_sample_size(*, paired_differences: dict[str, list[float]],
                         contrasts: list[ContrastSpec],
                         alpha: float = 0.05,
                         target_power: float = 0.8,
                         inflation: float = 1.25,
                         sensitivity_inflation: tuple[float, ...] = (1.0, 1.25,
                                                                    1.5, 2.0),
                         cost_per_campaign_s: float | None = None,
                         campaigns_per_block: int = 3,
                         ) -> dict[str, Any]:
    """Full sample-size recommendation (per contrast + sensitivity + budget)."""
    n_contrasts = max(1, len(contrasts))
    alpha_corrected = alpha / n_contrasts
    pilot: dict[str, Any] = {}
    estimated: dict[str, Any] = {}
    insufficient: list[str] = []

    for spec in contrasts:
        diffs = [float(d) for d in paired_differences.get(spec.name, [])]
        stats = summarise(diffs)
        pilot[spec.name] = {**stats, "seoi": spec.seoi,
                            "metric": spec.metric}
        if stats["n"] < MIN_BLOCKS_FOR_VARIANCE or stats["variance"] is None:
            insufficient.append(spec.name)
            continue
        n = blocks_required(variance_d=stats["variance"], seoi=spec.seoi,
                            alpha=alpha_corrected, power=target_power,
                            inflation=inflation)
        estimated[spec.name] = {"n_blocks": n, "variance_used": stats["variance"],
                                "inflation": inflation}

    grid: dict[str, list[dict[str, Any]]] = {}
    for spec in contrasts:
        rows: list[dict[str, Any]] = []
        base_var = pilot[spec.name]["variance"]
        for factor in sensitivity_inflation:
            for seoi in (spec.seoi, spec.seoi * 1.5, spec.seoi * 2.0):
                if base_var:
                    n = blocks_required(variance_d=base_var * factor,
                                        seoi=seoi, alpha=alpha_corrected,
                                        power=target_power, inflation=1.0)
                    rows.append({"variance_factor": factor, "seoi": seoi,
                                 "n_blocks": n,
                                 "source": "pilot_variance_inflated"})
                else:
                    # No pilot variance: report what each ASSUMED standardised
                    # effect would need (explicitly hypothetical, never used as
                    # a recommendation).
                    for d_std in (0.5, 0.8, 1.0):
                        n = int(math.ceil(
                            ((z_quantile(1 - alpha_corrected / 2)
                              + z_quantile(target_power)) ** 2)
                            / (d_std ** 2)))
                        rows.append({"standardised_effect": d_std,
                                     "n_blocks": n,
                                     "source": "hypothetical_standardised_effect"})
                    break
        grid[spec.name] = rows

    required = [v["n_blocks"] for v in estimated.values()]
    n_blocks_recommended = max(required) if required else None
    budget = None
    if n_blocks_recommended and cost_per_campaign_s:
        campaigns = n_blocks_recommended * campaigns_per_block
        budget = {"campaigns": campaigns,
                  "estimated_seconds": round(campaigns * cost_per_campaign_s, 1),
                  "estimated_hours": round(
                      campaigns * cost_per_campaign_s / 3600.0, 2),
                  "cost_per_campaign_s": cost_per_campaign_s,
                  "campaigns_per_block": campaigns_per_block}

    status = "evidence_insufficient" if insufficient else "estimated"
    return {
        "schema": "m4_3_sample_size_v1",
        "status": status,
        "design": "paired blocks (A/B/C within block), two-sided",
        "independent_unit": "block",
        "formula": ("n = ceil(((z_{1-alpha'/2} + z_{power})^2 * var_d * "
                    "inflation) / seoi^2)"),
        "alpha": alpha,
        "alpha_per_contrast": round(alpha_corrected, 8),
        "multiple_comparison_correction": (
            f"holm over {n_contrasts} contrasts; power computed with the "
            "conservative Bonferroni-equivalent alpha"),
        "target_power": target_power,
        "variance_inflation": inflation,
        "smallest_effect_of_interest_policy": (
            "pre-specified per contrast; the largest pilot effect is NEVER used"),
        "pilot_paired_differences": pilot,
        "per_contrast": estimated,
        "contrasts_without_variance": insufficient,
        "n_blocks_recommended": n_blocks_recommended,
        "sensitivity_grid": grid,
        "budget_estimate": budget,
        "notes": (
            ["pilot paired blocks are too few to estimate a trustworthy "
             "variance; only hypothetical standardised-effect rows are shown"
             ] if insufficient else
            ["variance is estimated from the pilot paired differences and "
             "then inflated conservatively; treat the recommendation as a "
             "planning figure, not a guarantee"]),
        "min_blocks_for_variance": MIN_BLOCKS_FOR_VARIANCE,
    }


__all__ = [
    "z_quantile", "MIN_BLOCKS_FOR_VARIANCE", "ContrastSpec", "blocks_required",
    "summarise", "estimate_sample_size",
]
