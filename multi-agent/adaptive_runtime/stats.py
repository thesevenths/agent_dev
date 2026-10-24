"""配对统计：门限 δ、t 临界值、p 值、功效反推样本量。

对应 V3 方案 §8：
    Δ_i = Q_cand(i) − Q_base(i)
    δ   = t_{1−α/2, n−1} · sd(Δ) / √n
    ACCEPT iff mean(Δ) > δ

为什么不用正态近似：n 常在 4–12，t 与 z 的差距不可忽略（n=5 时 2.776 vs 1.96）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Sequence


#: 双侧 95% 的 t 临界值（df=1..30 精确查表，之后用渐近式）
_T975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
    14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
    20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}
_Z975 = 1.959964


def _t_upper(alpha: float) -> float:
    """双边 α 对应的上分位（只支持常见 α，其余用正态近似）。"""
    if abs(alpha - 0.05) < 1e-9:
        return _Z975
    if abs(alpha - 0.10) < 1e-9:
        return 1.644854
    if abs(alpha - 0.01) < 1e-9:
        return 2.575829
    # 通用近似（Abramowitz-Stegun 26.2.23 的反函数）
    return _norm_ppf(1 - alpha / 2)


def _norm_ppf(p: float) -> float:
    """标准正态分位函数（Acklam 近似，误差 < 1.15e-9）。"""
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p 必须在 (0,1)")
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
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def t_crit(df: int, alpha: float = 0.05) -> float:
    """双侧 t 临界值。df 越大越接近正态分位。"""
    if df < 1:
        raise ValueError("df 必须 >= 1")
    if abs(alpha - 0.05) < 1e-9:
        if df in _T975:
            return _T975[df]
        # 渐近展开：t ≈ z + (z^3+z)/(4df) + (5z^5+16z^3+3z)/(96df^2)
        z = _Z975
        return z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)
    # 其余 α：数值反解（单调二分，范围 [0, 200]）
    z = _t_upper(alpha)
    approx = z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)
    return approx


# ---------------------------------------------------------------- beta 函数

def _betacf(a: float, b: float, x: float) -> float:
    """不完全 beta 的连分式（Lentz 算法，Numerical Recipes 6.4）。"""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-12:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """正则化不完全 beta 函数 I_x(a,b)。"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_pvalue(t: float, df: int) -> float:
    """双侧 p 值。t=0 → 1；|t|→∞ → 0。"""
    if df < 1:
        return 1.0
    if not math.isfinite(t):
        return 0.0
    x = df / (df + t * t)
    return max(0.0, min(1.0, betainc(df / 2.0, 0.5, x)))


# ---------------------------------------------------------------- 配对统计

@dataclass
class PairedResult:
    n: int
    mean: float
    sd: float
    se: float
    df: int
    t_stat: float
    p_value: float
    delta_crit: float
    positive: bool

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in d.items()}


def paired_stats(deltas: Sequence[float], alpha: float = 0.05) -> PairedResult:
    """对配对差值序列做单样本 t 检验，并给出判定门限 δ。

    退化情形（全部相同）：
      - 全 0 → delta_crit=0, positive=False
      - 全正 → delta_crit=0, positive=True（确定性改进，无噪声）
    """
    vals = [float(d) for d in deltas if d is not None and math.isfinite(float(d))]
    n = len(vals)
    if n == 0:
        return PairedResult(0, 0.0, 0.0, 0.0, 0, 0.0, 1.0, 0.0, False)
    mean = sum(vals) / n
    if n < 2:
        return PairedResult(n, mean, 0.0, 0.0, 0, 0.0, 1.0, 0.0, False)
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    df = n - 1
    if se == 0.0:
        # 零方差：结论完全由均值符号决定
        return PairedResult(n, mean, 0.0, 0.0, df, math.inf if mean > 0 else (-math.inf if mean < 0 else 0.0),
                            0.0 if mean != 0 else 1.0, 0.0, mean > 0)
    t_stat = mean / se
    p = t_pvalue(t_stat, df)
    delta_crit = t_crit(df, alpha) * se
    return PairedResult(n, mean, sd, se, df, t_stat, p, delta_crit, mean > delta_crit)


def delta_crit(n_tasks: int, n_reps: int, sd: float, alpha: float = 0.05) -> float:
    """配对 t 的检出门限 δ（Q 百分点）。**口径必须与 ``Evidence.paired()`` 一致**。

        δ = t_crit(T-1, α) · sd / sqrt(T · n)

    T = **不同任务数**，它才是自由度：rep 不是独立样本（同任务同代码的波动远小于
    跳任务波动，而且我们要回答的是「跳一个新任务还成立吗」）。
    n = 每个任务的重复次数，它只把**单个任务均值**的噪声按 1/n 收缩（n 条 run 取均值），
    不增加自由度。

    为什么专门写这个函数：旧版 validate() 把 ADAPTIVE_N 当成配对数算 δ（n=3 时
    报出 δ≈37 分），而实际跑法是 13 个 train 任务 × 3 次重复 ⇒ δ≈5.2 分。预告了不存
    在的门限会把能检出的改进报成检不出，是配置层面的自我欺骗。
    """
    T, n = max(1, int(n_tasks)), max(1, int(n_reps))
    if sd <= 0:
        return 0.0
    if T < 2:
        return math.inf        # 一个任务无法区分「普遍有效」与「恰好这一题」
    return t_crit(T - 1, alpha) * float(sd) / math.sqrt(T * n)


def required_n(sd: float, effect: float, power: float = 0.80, alpha: float = 0.05) -> int:
    """检出 ``effect`` 且达到给定把握所需的配对数。

    n ≈ ((z_{1−α/2} + z_power) · sd / effect)^2
    """
    if effect <= 0 or sd <= 0:
        return 1
    z_a = _t_upper(alpha)
    z_b = _norm_ppf(power)
    n = ((z_a + z_b) * sd / effect) ** 2
    return max(2, int(math.ceil(n)))


def mde(sd: float, n: int, power: float = 0.80, alpha: float = 0.05) -> float:
    """给定 sd 与 n 时，能达到指定把握的最小可检出效应（=δ 的实用版本）。"""
    if n < 2:
        return float("inf")
    df = n - 1
    se = sd / math.sqrt(n)
    # 双侧检验：δ 同时要覆盖 t 临界值与功效项
    return (t_crit(df, alpha) + _norm_ppf(power)) * se
