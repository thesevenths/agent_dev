"""比较层：把两臂的 score 配对成 Δ，产出 Evidence 与 comparison.json。

配对键是 ``(task_id, rep)``：同一任务同一重复次数的 base 与 candidate 才算一对。
任务难度不同、模型采样不同都会污染非配对比较，所以这里不做任何「取两组均值」的捷径。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Iterable

from .gate import Evidence
from .settings import CFG, Settings
from .stats import PairedResult, paired_stats


@dataclass
class PairRow:
    task_id: str
    rep: int
    base_Q: float
    cand_Q: float
    base_A: float = 0.0
    cand_A: float = 0.0
    #: 计划步数。``A = 通过门数 / 计划步数``，所以「少规划」本身就能抬高 A ——
    #: 这两个字段是让 gate 能识别「靠削减计划提分」的唯一依据。
    base_planned: int = 0
    cand_planned: int = 0

    @property
    def delta(self) -> float:
        return self.cand_Q - self.base_Q


@dataclass
class Comparison:
    pairs: list[PairRow] = field(default_factory=list)
    per_task: dict[str, dict] = field(default_factory=dict)
    missing_runs: int = 0
    timeout_runs: int = 0
    cost_base: int = 0
    cost_cand: int = 0
    judge_unstable_ratio: float = 0.0
    hard_failure_removed: bool = False
    add_novel_component: bool = False
    stats: dict = field(default_factory=dict)

    def evidence(self, cfg: Settings | None = None) -> Evidence:
        cfg = cfg or CFG
        deltas = [p.delta for p in self.pairs]
        dC = ((self.cost_cand - self.cost_base) / self.cost_base) if self.cost_base > 0 else 0.0
        regression = 0.0
        if self.per_task:
            bad = sum(1 for v in self.per_task.values()
                      if v.get("mean_delta", 0.0) < -cfg.regression_min_points)
            regression = bad / len(self.per_task)
        A_base = _mean([p.base_A for p in self.pairs]) if self.pairs else 0.0
        A_cand = _mean([p.cand_A for p in self.pairs]) if self.pairs else 0.0
        # 任务级聚合：同任务多 rep 先取均值，避免用 rep 数虚增自由度
        _by_task: dict[str, list[float]] = {}
        for p in self.pairs:
            _by_task.setdefault(p.task_id, []).append(p.delta)
        task_deltas = [_mean(v) for v in _by_task.values()]
        # 反刷分：A 的分母是「计划步数」，所以「把计划砍短」本身就能抬高 A。
        # 只统计「计划明显变短 **且** A 反而变高」的配对 —— 单看计划变短会误伤
        # 「因为效率提升真的不需要那么多步」的合法改进。
        undercut = 0
        for p in self.pairs:
            if p.base_planned <= 0 or p.cand_planned <= 0:
                continue
            shorter = p.cand_planned < p.base_planned * cfg.plan_undercut_ratio
            if shorter and p.cand_A > p.base_A + cfg.a_tol:
                undercut += 1
        n_with_plan = sum(1 for p in self.pairs if p.base_planned > 0 and p.cand_planned > 0)
        return Evidence(
            deltas=deltas,
            task_deltas=task_deltas,
            plan_undercut_rate=(undercut / n_with_plan) if n_with_plan else 0.0,
            A_base=A_base, A_cand=A_cand,
            dC=dC, regression_rate=regression, n_tasks=len(self.per_task),
            add_novel_component=self.add_novel_component,
            hard_failure_removed=self.hard_failure_removed,
            missing_runs=self.missing_runs, timeout_runs=self.timeout_runs,
            judge_unstable_ratio=self.judge_unstable_ratio,
            cost_base=self.cost_base, cost_cand=self.cost_cand,
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pairs"] = [{**asdict(p), "delta": round(p.delta, 4)} for p in self.pairs]
        return d


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def compare(base_scores: list[dict], cand_scores: list[dict],
            cfg: Settings | None = None) -> Comparison:
    """base/cand 均为 [{task_id, rep, Q, tokens?}, ...]。"""
    cfg = cfg or CFG
    bmap = {(s["task_id"], int(s.get("rep", 0))): s for s in base_scores}
    cmap = {(s["task_id"], int(s.get("rep", 0))): s for s in cand_scores}
    common = sorted(set(bmap) & set(cmap), key=lambda k: (k[0], k[1]))

    cmp = Comparison()
    for key in common:
        cmp.pairs.append(PairRow(task_id=key[0], rep=key[1],
                                 base_Q=float(bmap[key].get("Q", 0.0)),
                                 cand_Q=float(cmap[key].get("Q", 0.0)),
                                 base_A=float(bmap[key].get("A", 0.0)),
                                 cand_A=float(cmap[key].get("A", 0.0)),
                                 base_planned=int(bmap[key].get("planned", 0) or 0),
                                 cand_planned=int(cmap[key].get("planned", 0) or 0)))

    cmp.missing_runs = max(0, len(bmap) - len(common)) + max(0, len(cmap) - len(common))
    cmp.cost_base = sum(int(s.get("tokens", 0) or 0) for s in base_scores)
    cmp.cost_cand = sum(int(s.get("tokens", 0) or 0) for s in cand_scores)

    per: dict[str, list[float]] = {}
    for p in cmp.pairs:
        per.setdefault(p.task_id, []).append(p.delta)
    for task, ds in per.items():
        pr = paired_stats(ds, alpha=cfg.alpha)
        cmp.per_task[task] = {
            "n": len(ds),
            "mean_delta": round(pr.mean, 4),
            "sd": round(pr.sd, 4),
        }

    pr_all = paired_stats([p.delta for p in cmp.pairs], alpha=cfg.alpha)
    cmp.stats = pr_all.to_dict()
    cmp.stats["regression_rate"] = round(
        sum(1 for v in cmp.per_task.values()
            if v["mean_delta"] < -cfg.regression_min_points) / len(cmp.per_task), 4
    ) if cmp.per_task else 0.0
    cmp.stats["dC"] = round(((cmp.cost_cand - cmp.cost_base) / cmp.cost_base), 4) if cmp.cost_base else 0.0
    return cmp


def summarize(comparison: Comparison) -> str:
    """给报告用的一段文字（不参与判定）。"""
    st = comparison.stats
    lines = [
        f"配对 n={st.get('n', 0)}  mean(Δ)={st.get('mean', 0):+.2f}  "
        f"sd(Δ)={st.get('sd', 0):.2f}  δ={st.get('delta_crit', 0):.2f}  p={st.get('p_value', 1):.4f}",
        f"cost: base={comparison.cost_base:,} cand={comparison.cost_cand:,} "
        f"ΔC={st.get('dC', 0) * 100:+.1f}%",
        f"regression_rate={st.get('regression_rate', 0):.2f} "
        f"missing={comparison.missing_runs} timeout={comparison.timeout_runs}",
    ]
    return "\n".join(lines)


def paired_result(comparison: Comparison, cfg: Settings | None = None) -> PairedResult:
    return comparison.evidence(cfg).paired((cfg or CFG).alpha)
