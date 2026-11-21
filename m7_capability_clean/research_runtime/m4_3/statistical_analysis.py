"""M4.3-02 -- statistical analysis for the CONFIRMATORY paired-block design.

Reuses the project's existing, tested statistics (``evolution.statistics``)
instead of re-deriving a t distribution: two-sided Student-t p values, exact
t critical values (2.776 at df=4, never 1.96) and Holm step-down correction.

Only the confirmatory stage calls this. The pilot analysis is descriptive on
purpose and never imports a p value from here.

M4.3-02.1 P1-02: the multiple-comparison family is PRE-REGISTERED and FIXED.
A contrast with no computable statistic is NOT silently removed from the family
-- it stays as p=1.0 (least significant) so the correction denominator never
shrinks after results are seen. ``n_contrasts_in_family`` is the pre-registered
family size; ``n_contrasts_with_statistic`` reports how many were computable.
"""
from __future__ import annotations

import math
from statistics import mean, variance
from typing import Any, Sequence

from ..evolution.statistics import (
    holm_step_down, paired_confidence_interval, paired_p_value,
    student_t_crit_two_sided,
)

DEFAULT_ALPHA = 0.05


def cohens_dz(differences: Sequence[float]) -> float | None:
    """Standardised mean paired difference (dz = mean / sd)."""
    vals = [float(v) for v in differences]
    if len(vals) < 2:
        return None
    sd = math.sqrt(variance(vals))
    if sd == 0.0:
        return None
    return round(mean(vals) / sd, 6)


def paired_block_contrast(differences: Sequence[float],
                          *, alpha: float = DEFAULT_ALPHA) -> dict[str, Any]:
    """One paired contrast over BLOCKS (the independent unit)."""
    vals = [float(v) for v in differences]
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "p_value": None,
                "confidence_interval_95": None, "effect_size_dz": None,
                "verdict": "no_paired_blocks"}
    m = mean(vals)
    if n < 2:
        return {"n": n, "mean": round(m, 6), "sd": None, "p_value": None,
                "confidence_interval_95": None,
                "effect_size_dz": None, "verdict": "insufficient_blocks"}
    sd = math.sqrt(variance(vals))
    p = paired_p_value(m, sd, n)
    ci = paired_confidence_interval(m, sd, n, alpha=alpha)
    return {"n": n, "mean": round(m, 6), "sd": round(sd, 6),
            "p_value": round(p, 8), "confidence_interval_95": [round(ci[0], 6),
                                                              round(ci[1], 6)],
            "effect_size_dz": cohens_dz(vals),
            "t_critical": round(student_t_crit_two_sided(alpha, n - 1), 6),
            "df": n - 1,
            "verdict": "computed"}


def analyse_primary_contrasts(
        *, differences: dict[str, list[float]],
        alpha: float = DEFAULT_ALPHA,
        correction: str = "holm",
        family: Sequence[str] | None = None,
) -> dict[str, Any]:
    """All pre-registered contrasts with a family-wise correction.

    ``differences`` maps a contrast label (e.g. ``"C-B"``) to its per-block
    differences. ``family`` is the PRE-REGISTERED set of contrasts and is the
    correction denominator. It defaults to the keys of ``differences`` for
    backward compatibility, but the value report passes the FULL frozen family
    so a missing contrast can never shrink the correction (P1-02).

    A contrast with no data / no statistic stays explicitly present as
    ``p_value=None``, ``significant=False``; for the correction it is treated as
    p=1.0 (least significant), which keeps the family size fixed and can never
    make another contrast look more significant.
    """
    labels = list(family) if family is not None else list(differences)
    # Preserve order and drop accidental duplicates without changing the family.
    seen: set[str] = set()
    labels = [l for l in labels if not (l in seen or seen.add(l))]

    per: dict[str, Any] = {}
    for label in labels:
        if label in differences:
            per[label] = paired_block_contrast(differences[label], alpha=alpha)
        else:
            per[label] = {"n": 0, "mean": None, "sd": None, "p_value": None,
                          "confidence_interval_95": None,
                          "effect_size_dz": None, "verdict": "missing_contrast"}

    usable = [label for label in labels if per[label].get("p_value") is not None]
    # FIXED family: every pre-registered label contributes a p value; a missing
    # one contributes 1.0 so it stays in the denominator and cannot be dropped.
    family_p_values = [per[label]["p_value"] if label in usable else 1.0
                       for label in labels]
    if correction == "holm":
        corrected = holm_step_down(family_p_values, alpha=alpha)
    elif correction == "none":
        corrected = [{"significant": (p <= alpha), "effective_alpha": alpha,
                      "rank": i, "p_value": p, "n_compared": len(labels),
                      "method": "none"}
                     for i, p in enumerate(family_p_values)]
    else:
        raise ValueError(f"unknown multiple-comparison correction {correction!r}")
    for label, c in zip(labels, corrected):
        per[label]["correction"] = c
        # A contrast without a real statistic must NEVER be significant.
        per[label]["significant"] = bool(
            label in usable and c["significant"])
    missing = [label for label in labels if label not in usable]
    return {
        "alpha": alpha,
        "correction": correction,
        "n_contrasts_in_family": len(labels),
        "n_contrasts_with_statistic": len(usable),
        "family": labels,
        "missing_contrasts": missing,
        "contrasts": per,
        "labels": labels,
    }


__all__ = ["DEFAULT_ALPHA", "cohens_dz", "paired_block_contrast",
           "analyse_primary_contrasts"]
