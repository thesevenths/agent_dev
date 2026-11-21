"""噪声标定与样本量反推（P0 的核心交付）。

一张表就能说明为什么必须先标定：

    sd(Δ)=12, n=5   → δ≈14.9   「小于 15 分的改进都与噪声不可区分」
    sd(Δ)=12, n=71  → δ≈2.8    仍要 71 对才能达到 80% 把握

结论：**δ 只能靠降低 sd(Δ) 来缩小**，靠加样本量是拿钱买设计问题。
"""
from __future__ import annotations

import math
from typing import Iterable

from .io_util import now_iso, read_json, write_json
from .settings import CFG, PATHS, Settings
from .stats import mde, paired_stats, required_n


def pool_sd(groups: dict[str, list[float]]) -> dict:
    """组内合并标准差（pooled within-group sd）。"""
    num = 0.0
    den = 0
    per: dict[str, dict] = {}
    for task, vals in groups.items():
        vals = [float(v) for v in vals]
        if len(vals) >= 2:
            m = sum(vals) / len(vals)
            var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
            num += var * (len(vals) - 1)
            den += len(vals) - 1
            per[task] = {"n": len(vals), "mean": round(m, 3), "sd": round(math.sqrt(var), 3)}
        else:
            per[task] = {"n": len(vals), "mean": round(sum(vals) / len(vals), 3) if vals else 0.0,
                         "sd": 0.0, "insufficient": True}
    sd = math.sqrt(num / den) if den > 0 else 0.0
    return {"pooled_sd": round(sd, 4), "per_task": per, "df": den}


def null_diff_sd(groups: dict[str, list[float]]) -> float:
    """零改动下的配对差波动：同一任务内所有两两差值的标准差。

    这是 δ 真正的 σ：它把「同一 harness 跑两次也会差几分」直接量出来。
    """
    diffs: list[float] = []
    for vals in groups.values():
        vals = [float(v) for v in vals]
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                diffs.append(vals[i] - vals[j])
    if len(diffs) < 2:
        return 0.0
    m = sum(diffs) / len(diffs)
    var = sum((d - m) ** 2 for d in diffs) / (len(diffs) - 1)
    return math.sqrt(var)


def n_pairs(groups: dict[str, list[float]]) -> int:
    """可用的配对差个数：每任务 C(k,2)，只算跑满 ≥2 次的任务。"""
    return sum(len(v) * (len(v) - 1) // 2 for v in groups.values() if len(v) >= 2)


def validity(groups: dict[str, list[float]], sd_null: float,
             cfg: Settings | None = None) -> list[str]:
    """标本报表能不能信 —— 返回「不能信」的原因列表（空 = 可信）。

    为什么不能只看 ``sd_null < target``：sd=0 在数学上是「噪声为零」，在现实里几乎总是
    **样本退化**（所有 run 都因为同一个基础设施故障拿到同一个地板分）。把那种报表
    当标定，δ 会算成 0，于是「任何正数的 ΔQ 都显著」—— 那是把判分口径交还给噪声。
    """
    cfg = cfg or CFG
    bad: list[str] = []
    tasks_ge2 = sum(1 for v in groups.values() if len(v) >= 2)
    if tasks_ge2 < cfg.min_calib_tasks:
        bad.append(f"跑满 ≥2 次的任务只有 {tasks_ge2} 个，少于最低要求 {cfg.min_calib_tasks}")
    if n_pairs(groups) < cfg.min_tasks_for_t:
        bad.append(f"配对差只有 {n_pairs(groups)} 个，不足以估计 sd(Δ)")
    if sd_null == 0.0:
        bad.append("sd(Δ)=0：零方差不是低噪声，而是样本退化（多半是所有 run 都挂了或都是假跑）")
    return bad


def summarize(groups: dict[str, list[float]], cfg: Settings | None = None) -> dict:
    """从「每任务重复 k 次的 Q」得到一份标定报告。"""
    cfg = cfg or CFG
    pooled = pool_sd(groups)
    sd_null = null_diff_sd(groups)
    n_eff = sum(len(v) for v in groups.values()) - len(groups)
    out = {
        "created_at": now_iso(),
        "generated_by": "adaptive_runtime.calibrate",
        "k": min((len(v) for v in groups.values()), default=0),
        "tasks": len(groups),
        "n_observations": sum(len(v) for v in groups.values()),
        "sd_run": pooled["pooled_sd"],
        "df": pooled["df"],
        "sd_diff_null": round(sd_null, 4),
        "sd_diff_theory": round(pooled["pooled_sd"] * math.sqrt(2), 4),
        "per_task": pooled["per_task"],
        "n_pairs": n_pairs(groups),
        "target": cfg.sd_diff_target,
        "validity": validity(groups, sd_null, cfg),
        "passed": sd_null < cfg.sd_diff_target and not validity(groups, sd_null, cfg),
        "mde_at_default_n": {
            "n": cfg.n_default,
            "effect_80pct": round(mde(sd_null, cfg.n_default, cfg.power, cfg.alpha), 2),
        },
        "required_n_for_4pts": required_n(sd_null, 4.0, cfg.power, cfg.alpha),
        "required_n_for_8pts": required_n(sd_null, 8.0, cfg.power, cfg.alpha),
    }
    if n_eff <= 0:
        out["passed"] = False
        out["warning"] = "有效自由度不足（请每任务至少跑 2 次）"
    return out


def passes(report: dict, cfg: Settings | None = None) -> bool:
    cfg = cfg or CFG
    if report.get("validity"):
        return False                    # 报表自己说了不可信，就不必再看 sd
    return bool(report.get("passed")) and float(report.get("sd_diff_null", 1e9)) < cfg.sd_diff_target


def save(report: dict, path=None) -> None:
    write_json(path or PATHS.calibration, report)


def load(path=None) -> dict:
    return read_json(path or PATHS.calibration, default={}) or {}


def suggest_n(sd: float, effect: float = 4.0, cfg: Settings | None = None) -> int:
    cfg = cfg or CFG
    return required_n(sd, effect, cfg.power, cfg.alpha)


def table(sd: float, ns: Iterable[int] = (5, 10, 25, 50), cfg: Settings | None = None) -> list[dict]:
    """固定 sd，列出各 n 下的 δ（用于报告里解释「为什么是 n=8」）。"""
    cfg = cfg or CFG
    from .stats import t_crit
    rows = []
    for n in ns:
        delta = t_crit(max(1, int(n) - 1), cfg.alpha) * sd / math.sqrt(n)
        rows.append({"n": int(n), "delta_crit": round(delta, 3), "power": cfg.power})
    return rows
