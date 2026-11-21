"""ARR 1.1 / M3.0 -- dependency-free statistical primitives for the DOMAIN gate.

Why this module exists
----------------------
M2.7 computed the paired-improvement confidence interval with the NORMAL
critical value ``1.96``. That is only valid asymptotically; with the formal
protocol's ``n=5`` paired seeds (df=4) the correct two-sided 95% Student-t
critical value is ``2.776``. Using 1.96 systematically UNDERSTATES the
uncertainty and can produce false-positive ACCEPTs (exactly what happened to
two of the three M2.7 SciNet++ candidates).

M3.0 also adds family-wise multiple-comparison control: a round evaluates
several candidates (e.g. three learning rates) and picks the best, which
inflates the false-positive rate unless the significance level is corrected.
We use Holm's step-down procedure.

Implementation constraint
-------------------------
The engine-free CI installs ONLY ``PyYAML`` + ``pydantic``. There is deliberately
NO numpy / scipy / pandas. Everything below is pure Python standard library:

* regularized incomplete beta function via Lentz's continued fraction
  (Numerical Recipes ``betacf`` / ``betai``);
* ``P(|T_df| > t) = I_{df/(df+t^2)}(df/2, 1/2)``;
* the critical value is obtained by monotone bisection on that tail probability.

All functions are deterministic and side-effect free, so a verdict can always
be recomputed from the persisted evidence (no hidden RNG, no hidden state).
"""
from __future__ import annotations

import math
from typing import Any, Sequence

#: Family-wise significance level for the frozen M3.0 statistical protocol.
#: This is a PROTOCOL CONSTANT, not a tunable knob: exposing a CLI flag that
#: could raise it would let a caller manufacture ACCEPTs. Callers may only make
#: it stricter (Holm always does).
DEFAULT_ALPHA = 0.05


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction expansion of the incomplete beta (Lentz's method)."""
    max_iter = 300
    eps = 3.0e-14
    fpmin = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function ``I_x(a, b)`` in ``[0, 1]``."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        value = front * _betacf(a, b, x) / a
    else:
        value = 1.0 - front * _betacf(b, a, 1.0 - x) / b
    return min(1.0, max(0.0, value))


def student_t_two_sided_p(t: float, df: int) -> float:
    """Two-sided tail probability ``P(|T_df| >= |t|)``.

    Uses the exact identity ``P(|T_df| > t) = I_{df/(df+t^2)}(df/2, 1/2)``.
    """
    if df <= 0:
        raise ValueError("df must be a positive integer")
    t = abs(float(t))
    if math.isinf(t) or math.isnan(t):
        return 0.0 if math.isinf(t) else 1.0
    x = float(df) / (float(df) + t * t)
    return _betainc(float(df) / 2.0, 0.5, x)


def student_t_crit_two_sided(alpha: float = DEFAULT_ALPHA, df: int = 4) -> float:
    """Two-sided Student-t critical value whose tail probability is ``alpha``.

    Monotone bisection (``p(t)`` is strictly decreasing in ``t >= 0``). At
    ``df=4, alpha=0.05`` this returns ``2.776445...`` -- NOT 1.96.
    """
    if not (math.isfinite(alpha) and 0.0 < alpha < 1.0):
        raise ValueError("alpha must lie strictly inside (0, 1)")
    if df <= 0:
        raise ValueError("df must be a positive integer")
    lo, hi = 0.0, 1.0
    while student_t_two_sided_p(hi, df) > alpha and hi < 1.0e12:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if student_t_two_sided_p(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1.0e-13:
            break
    return 0.5 * (lo + hi)


def paired_t_statistic(mean: float, std: float, n: int) -> float | None:
    """Paired t statistic ``mean / (std / sqrt(n))``, or ``None`` if undefined.

    ``std == 0`` is NOT a missing statistic: every paired difference is
    identical, so the improvement is deterministic (|t| = inf, p = 0). M2.7
    wrongly treated ``std == 0`` as "skip the CI check"; here it is explicit.
    """
    if n < 2:
        return None
    if std == 0.0:
        if mean > 0.0:
            return math.inf
        if mean < 0.0:
            return -math.inf
        return 0.0
    return float(mean) / (float(std) / math.sqrt(n))


def paired_p_value(mean: float, std: float, n: int) -> float:
    """Two-sided p value for one paired-improvement sample."""
    t = paired_t_statistic(mean, std, n)
    if t is None:
        return 1.0
    if math.isinf(t):
        return 0.0
    return student_t_two_sided_p(t, n - 1)


def paired_confidence_interval(mean: float, std: float, n: int,
                               alpha: float = DEFAULT_ALPHA
                               ) -> tuple[float, float]:
    """Two-sided ``1-alpha`` Student-t CI for the mean paired improvement."""
    crit = student_t_crit_two_sided(alpha, n - 1)
    se = float(std) / math.sqrt(n)
    return (mean - crit * se, mean + crit * se)


def holm_step_down(p_values: Sequence[float],
                   alpha: float = DEFAULT_ALPHA) -> list[dict[str, Any]]:
    """Holm-Bonferroni step-down correction for one FAMILY of candidates.

    Returns one audit dict per input (same order), with:
      ``significant``      -- whether the null "no improvement" is rejected;
      ``effective_alpha``  -- the (stricter) per-rank alpha, or 0.0 if the
                              step-down procedure already stopped;
      ``rank``             -- 0-based rank by ascending p value;
      ``p_value``          -- the unadjusted two-sided p value;
      ``n_compared``       -- family size m;
      ``method``           -- ``holm-step-down``.

    Step-down semantics: order p ascending; candidate at rank k is compared to
    ``alpha/(m-k)``. The first failure stops the procedure -- that candidate and
    every less-significant one remain non-significant. A non-finite p value is
    treated as 1.0 (least significant), never as a pass.
    """
    if not (math.isfinite(alpha) and 0.0 < alpha < 1.0):
        raise ValueError("alpha must lie strictly inside (0, 1)")
    m = len(p_values)
    if m == 0:
        return []
    clean = [p if (p is not None and math.isfinite(p)) else 1.0
             for p in p_values]
    clean = [min(1.0, max(0.0, float(p))) for p in clean]
    order = sorted(range(m), key=lambda i: (clean[i], i))
    out: list[dict[str, Any]] = [{} for _ in range(m)]
    stopped = False
    for k, i in enumerate(order):
        effective = alpha / (m - k)
        passed = (not stopped) and clean[i] <= effective
        if not passed:
            stopped = True
        out[i] = {
            "significant": bool(passed),
            "effective_alpha": effective if passed else 0.0,
            "rank": k,
            "p_value": clean[i],
            "n_compared": m,
            "method": "holm-step-down",
        }
    return out


__all__ = [
    "DEFAULT_ALPHA",
    "student_t_two_sided_p",
    "student_t_crit_two_sided",
    "paired_t_statistic",
    "paired_p_value",
    "paired_confidence_interval",
    "holm_step_down",
]
