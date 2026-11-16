"""M4.3-02 -- statistical analysis for the CONFIRMATORY paired-block design.

Reuses the project's existing, tested statistics (``evolution.statistics``)
instead of re-deriving a t distribution: two-sided Student-t p values, exact
t critical values (2.776 at df=4, never 1.96) and Holm step-down correction.

Only the confirmatory stage calls this. The pilot analysis is descriptive on
purpose and never imports a p value from here.
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


def analyse_primary_contrasts(*, differences: dict[str, list[float]],
                              alpha: float = DEFAULT_ALPHA,
                              correction: str = "holm") -> dict[str, Any]:
    """All PRIMARY contrasts with a family-wise correction.

    ``differences`` maps a contrast label (e.g. ``"C-B"``) to its per-block
    differences. The family is corrected as a whole; a contrast with no data
    stays explicitly missing instead of being silently dropped from the family.
    """
    labels = list(differences)
    per: dict[str, Any] = {label: paired_block_contrast(differences[label],
                                                        alpha=alpha)
                           for label in labels}
    usable = [label for label in labels if per[label].get("p_value") is not None]
    p_values = [per[label]["p_value"] for label in usable]
    if correction == "holm":
        corrected = holm_step_down(p_values, alpha=alpha)
    elif correction == "none":
        corrected = [{"significant": (p <= alpha), "effective_alpha": alpha,
                      "rank": i, "p_value": p, "n_compared": len(p_values),
                      "method": "none"} for i, p in enumerate(p_values)]
    else:
        raise ValueError(f"unknown multiple-comparison correction {correction!r}")
    for label, c in zip(usable, corrected):
        per[label]["correction"] = c
        per[label]["significant"] = bool(c["significant"])
    for label in labels:
        if label not in usable:
            per[label]["significant"] = False
            per[label]["correction"] = None
    return {"alpha": alpha, "correction": correction,
            "n_contrasts_in_family": len(usable),
            "contrasts": per,
            "labels": labels}


__all__ = ["DEFAULT_ALPHA", "cohens_dz", "paired_block_contrast",
           "analyse_primary_contrasts"]
