"""ARR-03 statistical protocol: task-level units (never seeds), paired tests,
small-sample intervals, permutation, cluster bootstrap, Holm correction."""
from __future__ import annotations

import math
import random
import statistics
from typing import Any, Sequence

from research_runtime.evolution.statistics import (
    paired_p_value, student_t_crit_two_sided,
)


def mean(vals: Sequence[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def paired_stats(diffs: Sequence[float], *, alpha: float = 0.05,
                 n_boot: int = 4000, n_perm: int = 4000,
                 seed: int = 20261122) -> dict[str, Any]:
    """Task-level paired inference. Units are regimes, seeds already averaged."""
    vals = [float(d) for d in diffs]
    n = len(vals)
    m = mean(vals)
    if n < 2:
        return {"n": n, "mean": m, "p": 1.0, "ci95": [0.0, 0.0],
                "permutation_p": 1.0, "bootstrap_ci95": [0.0, 0.0],
                "dz": None, "sd": 0.0}
    sd = statistics.stdev(vals)
    p = paired_p_value(m, sd, n)
    crit = student_t_crit_two_sided(alpha=alpha, df=n - 1)
    margin = crit * sd / math.sqrt(n)
    dz = m / sd if sd > 0 else None

    rng = random.Random(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        boots.append(mean([vals[rng.randrange(n)] for _ in range(n)]))
    boots.sort()
    lo = int(alpha / 2 * n_boot)
    hi = min(n_boot - 1, int((1 - alpha / 2) * n_boot))

    obs = abs(m)
    ge = 0
    for _ in range(n_perm):
        perm = [v * (1 if rng.random() < 0.5 else -1) for v in vals]
        if abs(mean(perm)) >= obs - 1e-15:
            ge += 1
    perm_p = (ge + 1) / (n_perm + 1)

    return {"n": n, "mean": m, "sd": sd, "p": p,
            "ci95": [m - margin, m + margin],
            "permutation_p": perm_p,
            "bootstrap_ci95": [boots[lo], boots[hi]],
            "dz": dz}


def holm(p_values: Sequence[float], alpha: float = 0.05) -> list[dict[str, Any]]:
    """Holm step-down correction; returns adjusted p and significance in input order."""
    indexed = sorted((float(p), i) for i, p in enumerate(p_values))
    m = len(indexed)
    adjusted = [0.0] * m
    running = 0.0
    for rank, (p, idx) in enumerate(indexed):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[idx] = running
    return [{"p": p, "holm_p": adjusted[i], "significant": adjusted[i] <= alpha}
            for i, p in enumerate(p_values)]


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, center - half), min(1.0, center + half)]


__all__ = ["mean", "paired_stats", "holm", "wilson"]
