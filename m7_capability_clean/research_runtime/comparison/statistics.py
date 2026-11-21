"""P1-05: paired-seed statistics, DEPENDENCY-FREE (standard library only).

Method: Student's t confidence interval on the paired seed deltas::

    d_i = B_i - C_i   (minimize: lower-is-better)  positive = improvement
    d_i = C_i - B_i   (maximize: higher-is-better) positive = improvement
    CI = mean(d) +- t_{1-alpha/2, n-1} * std(d, ddof=1) / sqrt(n)

Why no SciPy / NumPy dependency
-------------------------------
The Agent interpreter already has SciPy (via scikit-learn), but requirements
declares no direct dependency and adding a large numeric stack to a generic
research runtime is not justified for a single t quantile. The regularized
incomplete beta function below computes the exact two-sided t p-value and the
exact critical value; tests cross-check it against SciPy when available.

Assumptions (repeated in every report's notes)
---------------------------------------------
* Paired seeds are INDEPENDENT training runs (same seed => same initialisation
  for both arms), so a paired-delta interval measures TRAINING-SEED
  uncertainty. It does NOT measure per-sample prediction uncertainty.
* The deltas are approximately normal. With n < 10 a t interval is only
  approximate; that is why the decision gate requires >= min_paired_seeds.
* NO bootstrap. Resampling 3 seeds cannot create information, and a
  sample-level paired bootstrap answers a DIFFERENT question (it needs
  per-sample predictions with matching ids). It is deliberately left for later.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from .exceptions import ComparisonInputError

INTERVAL_METHOD = "student_t_paired_seed"

# --------------------------------------------------------------------------- #
# Regularized incomplete beta (Numerical Recipes continued fraction)
# --------------------------------------------------------------------------- #
def _betacf(a: float, b: float, x: float) -> float:
    max_iter, eps, fpmin = 300, 3.0e-14, 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
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


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b), clamped to [0, 1]."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    try:
        bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                      + a * math.log(x) + b * math.log1p(-x))
    except (ValueError, OverflowError):
        return 0.0
    if x < (a + 1.0) / (a + b + 2.0):
        value = bt * _betacf(a, b, x) / a
    else:
        value = 1.0 - bt * _betacf(b, a, 1.0 - x) / b
    return min(1.0, max(0.0, value))


def student_t_two_sided_p(t: float, df: int) -> float:
    """Two-sided p-value of Student's t. For t=inf this is 0."""
    if df <= 0:
        return 1.0
    if not math.isfinite(t):
        return 0.0
    x = df / (df + t * t)
    return _betai(df / 2.0, 0.5, x)


def student_t_critical(df: int, alpha: float) -> float:
    """t_{1-alpha/2, df}: the two-sided critical value for significance alpha."""
    if df <= 0 or not (0.0 < alpha < 1.0):
        raise ComparisonInputError("invalid df/alpha for t critical value")
    lo, hi = 0.0, 1.0e6
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if student_t_two_sided_p(mid, df) > alpha:
            lo = mid           # p still too large => need a larger t
        else:
            hi = mid
    return (lo + hi) / 2.0


# --------------------------------------------------------------------------- #
# Paired deltas and their summary
# --------------------------------------------------------------------------- #
def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComparisonInputError(f"{label} must be a finite number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ComparisonInputError(f"{label} must be finite, got {value!r}")
    return number


def normalize_deltas(baseline_values: Sequence[Any],
                      candidate_values: Sequence[Any],
                      direction: str) -> list[float]:
    """Direction-normalised improvement: POSITIVE always means BETTER."""
    if direction not in ("minimize", "maximize"):
        raise ComparisonInputError(
            f"unknown metric_direction {direction!r}; refusing to guess")
    if len(baseline_values) != len(candidate_values):
        raise ComparisonInputError("paired observations must have equal length")
    out: list[float] = []
    for index, (b_raw, c_raw) in enumerate(zip(baseline_values, candidate_values)):
        b = _finite(b_raw, f"baseline_values[{index}]")
        c = _finite(c_raw, f"candidate_values[{index}]")
        out.append(b - c if direction == "minimize" else c - b)
    return out


@dataclass(frozen=True)
class PairedStatistics:
    n: int
    mean_delta: float
    median_delta: float
    std_delta: float | None
    confidence_interval: tuple[float, float] | None
    alpha: float
    interval_method: str = INTERVAL_METHOD
    t_statistic: float | None = None
    p_value_two_sided: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def confidence_level(self) -> float:
        return 1.0 - self.alpha


@dataclass(frozen=True)
class HolmEntry:
    index: int
    rank: int
    p_value: float
    adjusted_p: float
    adjusted_alpha: float          # alpha/(m-rank): CI significance level
    familywise_alpha: float        # the alpha the adjusted p is compared to

    @property
    def significant(self) -> bool:
        # Standard Holm step-down: the monotone adjusted p is compared with
        # the FAMILY-WISE alpha, not with the per-rank alpha (that would be a
        # double correction). The per-rank alpha widens the reported CI.
        return self.adjusted_p < self.familywise_alpha


@dataclass(frozen=True)
class HolmResult:
    family_size: int
    familywise_alpha: float
    method: str
    entries: list[HolmEntry]        # aligned with the input p-value order


def holm_adjust(p_values: Sequence[float], *,
                familywise_alpha: float = 0.05) -> HolmResult:
    """Holm step-down control of the family-wise error rate.

    Rank j (0-based, ascending p) is tested at alpha/(m-j). ``adjusted_p`` is
    the monotone running maximum of (m-j)*p_j, clipped to 1. Callers use
    ``adjusted_alpha`` as the significance level for that hypothesis's
    confidence interval, and ``significant`` as the explicit gate.
    """
    if not (0.0 < familywise_alpha < 1.0):
        raise ComparisonInputError("familywise_alpha must be in (0, 1)")
    m = len(p_values)
    if m == 0:
        return HolmResult(0, familywise_alpha, "holm", [])
    cleaned = []
    for i, p in enumerate(p_values):
        value = 1.0 if p is None else _finite(p, f"p_values[{i}]")
        cleaned.append(min(1.0, max(0.0, value)))
    order = sorted(range(m), key=lambda i: cleaned[i])
    entries: list[HolmEntry | None] = [None] * m
    running = 0.0
    for rank, index in enumerate(order):
        multiplier = m - rank
        running = min(1.0, max(running, multiplier * cleaned[index]))
        entries[index] = HolmEntry(
            index=index, rank=rank, p_value=cleaned[index],
            adjusted_p=running,
            adjusted_alpha=familywise_alpha / multiplier,
            familywise_alpha=familywise_alpha,
        )
    # ``entries`` is indexed by the ORIGINAL p-value order; every slot is set.
    return HolmResult(m, familywise_alpha, "holm",
                      [e for e in entries if e is not None])  # type: ignore[misc]


class PairedStatisticsEngine:
    """Injectable statistics provider (keeps Comparator free of math policy)."""

    normalize = staticmethod(normalize_deltas)
    holm_adjust = staticmethod(holm_adjust)

    def summarize(self, deltas: Sequence[Any], *,
                  alpha: float = 0.05) -> PairedStatistics:
        if not (0.0 < alpha < 1.0):
            raise ComparisonInputError("alpha must be in (0, 1)")
        values = [_finite(d, f"paired_deltas[{i}]")
                  for i, d in enumerate(deltas)]
        n = len(values)
        if n == 0:
            return PairedStatistics(
                n=0, mean_delta=0.0, median_delta=0.0, std_delta=None,
                confidence_interval=None, alpha=alpha,
                notes=["no valid paired observations"],
            )
        mean = sum(values) / n
        ordered = sorted(values)
        median = (ordered[n // 2] if n % 2
                  else (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0)
        notes: list[str] = []
        if n < 2:
            notes.append("n < 2: no confidence interval or p-value is computed")
            return PairedStatistics(
                n=n, mean_delta=mean, median_delta=median, std_delta=None,
                confidence_interval=None, alpha=alpha,
                notes=notes,
            )

        variance = sum((d - mean) ** 2 for d in values) / (n - 1)
        std = math.sqrt(variance)
        if std == 0.0:
            # Every seed moved identically. The interval is a point; the
            # p-value is 0 for a non-zero effect and 1 for a null effect.
            notes.append("zero variance across paired seeds: point interval")
            p_value = 0.0 if mean != 0.0 else 1.0
            return PairedStatistics(
                n=n, mean_delta=mean, median_delta=median, std_delta=std,
                confidence_interval=(mean, mean), alpha=alpha,
                t_statistic=None, p_value_two_sided=p_value, notes=notes,
            )

        standard_error = std / math.sqrt(n)
        critical = student_t_critical(n - 1, alpha)
        half_width = critical * standard_error
        t_stat = mean / standard_error
        notes.append(
            "Student t interval on paired seed deltas; assumes independent "
            "paired runs and approximately normal deltas. It quantifies "
            "training-seed uncertainty, NOT per-sample prediction uncertainty."
        )
        if n < 10:
            notes.append("small n: the t interval is approximate")
        return PairedStatistics(
            n=n, mean_delta=mean, median_delta=median, std_delta=std,
            confidence_interval=(mean - half_width, mean + half_width),
            alpha=alpha, t_statistic=t_stat,
            p_value_two_sided=student_t_two_sided_p(abs(t_stat), n - 1),
            notes=notes,
        )
