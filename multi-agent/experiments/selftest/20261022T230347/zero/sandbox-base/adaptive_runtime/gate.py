"""判定层：非补偿 Gate（Gate A 在评分前，Gate B 在评分后）。

核心原则：**硬约束不可补偿**。行为指标 ΔB 不出现在这个文件里 ——
它是归因证据，不是准入依据。一旦 ΔB 获得准入权，闭环会优先学会堆砌来源。

相对 V2 的两处修正（都是实测会出错的地方）
------------------------------------------
1. A_FLOOR 不能是绝对门槛。若 base 的 A=0.60，绝对门槛 0.85 会把「A 从 0.60 提升到
   0.80」这种正是需要的候选判定为不合格。改为 floor = min(A_FLOOR, A_base) − A_TOL。
2. 明显变差必须直接判 REGRESSION。V2 的伪代码只查 regression_rate，而一个
   「每个任务都略差」的候选会先落进 WITHIN_NOISE 分支，将「变差」误判为「无差异」。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any

from .settings import CFG, Settings
from .stats import PairedResult, paired_stats


class Reason(str, Enum):
    FROZEN_CORE_VIOLATION = "frozen_core_violation"
    BENCHMARK_LEAKAGE = "benchmark_leakage"
    INERT_MECHANISM = "inert_mechanism"
    COST_CAP = "cost_cap"
    EDIT_BUDGET = "edit_budget"
    INCOMPLETE_EVIDENCE = "incomplete_evidence"
    CONTRACT_FLOOR = "contract_floor"
    JUDGE_UNSTABLE = "judge_unstable"
    REGRESSION = "regression"
    WITHIN_NOISE = "within_noise"
    COST_RULE = "cost_rule"


class Accept(str, Enum):
    VALIDATED = "VALIDATED"
    CHEAPER_EQUAL = "CHEAPER_EQUAL"
    NOVELTY = "NOVELTY"
    HARD_FAILURE_REMOVED = "HARD_FAILURE_REMOVED"
    GATE_A_PASS = "GATE_A_PASS"


#: 看着像「质量旋钮」、实际不改变任何行为路径的惰性机制。
#: 调整它们的收益恒为 0，应直接拒绝（也是 Critic 的第一个真实用例）。
INERT_TARGETS = frozenset({
    "AGENT_CRITIC_MIN_LEN",          # 默认 4 个字符 = 没有门槛（代码注释明确「不是质量门槛」）
    "AGENT_TRUNCATED_SALVAGE",       # 开关，不是阈值
    "AGENT_CRITIC_LLM",              # 开关
    "AGENT_FINAL_ADJUDICATE",        # 开关
})


@dataclass
class Decision:
    decision: str                      # ACCEPT | REJECT
    reason: str                        # Reason.value 或 Accept.value
    gate_trace: list[dict] = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.decision == "ACCEPT"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Evidence:
    deltas: list[float]
    #: 每个**任务**的均值 delta（同一任务的多 rep 已聚合）。t 检验必须用它。
    #: 理由：rep 之间不是独立样本 —— temp=0 的确定性后端会让同任务两次跑出完全相同的
    #: Q，sd=0 → δ=0，于是 n=2 就判「显著正收益」。真正要泛化的是「换一个新任务还成不成立」，
    #: 所以方差必须取**任务间**方差，自由度也必须按任务数算。
    task_deltas: list[float] = field(default_factory=list)
    A_base: float = 0.0
    A_cand: float = 0.0
    dC: float = 0.0
    regression_rate: float = 0.0
    n_tasks: int = 0
    add_novel_component: bool = False
    hard_failure_removed: bool = False
    missing_runs: int = 0
    timeout_runs: int = 0
    judge_unstable_ratio: float = 0.0
    #: 「计划明显变短 且 A 反而上升」的配对占比。A 的分母是计划步数，
    #: 所以这条路能让 Q 上升而不产生任何真实产出 —— 必须单独拦。
    plan_undercut_rate: float = 0.0
    cost_base: int = 0
    cost_cand: int = 0

    def paired(self, alpha: float = 0.05) -> PairedResult:
        return paired_stats(self.task_deltas or self.deltas, alpha=alpha)

    @property
    def n_pairs(self) -> int:
        return len(self.deltas)


def _trace(rows: list[dict], check: str, ok: bool, **value: Any) -> None:
    rows.append({"check": check, "pass": bool(ok), **value})


# ---------------------------------------------------------------- Gate A

def gate_a(cand: Any, frozen_violations: list[Any] | None = None,
           leakage_hits: list[str] | None = None,
           cfg: Settings | None = None) -> Decision:
    """评分前拦截：冻结、泄漏、惰性机制、预算、改动幅度。"""
    cfg = cfg or CFG
    rows: list[dict] = []
    violations = list(frozen_violations or [])
    leaks = list(leakage_hits or [])

    _trace(rows, "frozen_core", not violations, violations=[str(v) for v in violations])
    if violations:
        return Decision("REJECT", Reason.FROZEN_CORE_VIOLATION.value, rows,
                        {"violations": [getattr(v, "to_dict", lambda: str(v))() for v in violations]})

    _trace(rows, "leakage", not leaks, hits=leaks)
    if leaks:
        return Decision("REJECT", Reason.BENCHMARK_LEAKAGE.value, rows, {"hits": leaks})

    target = getattr(cand, "target", "") or ""
    inert = target in INERT_TARGETS
    _trace(rows, "not_inert", not inert, target=target)
    if inert:
        return Decision("REJECT", Reason.INERT_MECHANISM.value, rows,
                        {"note": "该参数是惰性机制/开关，调整它不会改变行为"})

    est = int(getattr(cand, "est_cost", 0) or 0)
    cap = min(cfg.cost_cap_cand, cfg.cost_cap_round)
    _trace(rows, "cost_cap", est <= cap, est_cost=est, cap=cap)
    if est > cap:
        return Decision("REJECT", Reason.COST_CAP.value, rows, {"est_cost": est, "cap": cap})

    level = int(getattr(cand, "level", 1) or 1)
    edits = int(getattr(cand, "edits", 1) or 1)
    budget = cfg.edit_budget_p0
    ok_budget = level <= cfg.max_edit_level and edits <= budget
    _trace(rows, "edit_budget", ok_budget, level=level, edits=edits,
           max_level=cfg.max_edit_level, budget=budget)
    if not ok_budget:
        return Decision("REJECT", Reason.EDIT_BUDGET.value, rows,
                        {"level": level, "edits": edits, "max_level": cfg.max_edit_level,
                         "budget": budget})

    return Decision("ACCEPT", Accept.GATE_A_PASS.value, rows)


# ---------------------------------------------------------------- Gate B

def gate_b(ev: Evidence, cfg: Settings | None = None) -> Decision:
    """评分后判定。顺序即优先级：硬约束 → 主目标 → 次要约束。"""
    cfg = cfg or CFG
    rows: list[dict] = []

    # ---- 1. 证据完整性（缺一次 run 就整轮作废，不插补）----
    incomplete = (ev.missing_runs + ev.timeout_runs) > 0
    _trace(rows, "evidence_complete", not incomplete,
           missing=ev.missing_runs, timeout=ev.timeout_runs)
    if incomplete:
        return Decision("REJECT", Reason.INCOMPLETE_EVIDENCE.value, rows,
                        {"missing": ev.missing_runs, "timeout": ev.timeout_runs})

    # ---- 1b. 任务数下限（少于 2 个任务时 t 检验无定义）----
    # 为什么这条必须存在：n_tasks=1 时 df=0、δ=0，「均值>0」就等价于「显著为正」，
    # 于是「拿一个任务跑一次」就能验收任何改动。这不是统计功效不足，是逻辑上不成立 ——
    # 我们要的是「换任务还成立」，单任务证据原理上无法回答这个问题。
    n_tasks = len(ev.task_deltas) if ev.task_deltas else ev.n_tasks
    ok_min_tasks = n_tasks >= cfg.min_tasks_for_t
    _trace(rows, "min_tasks", ok_min_tasks, n_tasks=n_tasks, need=cfg.min_tasks_for_t)
    if not ok_min_tasks:
        return Decision("REJECT", Reason.INCOMPLETE_EVIDENCE.value, rows,
                        {"n_tasks": n_tasks, "need_min_tasks": cfg.min_tasks_for_t,
                         "note": "任务数不足，无法区分「对该任务有效」与「普遍有效」"})

    # ---- 2. 契约地板（相对 base，不是绝对值）----
    floor = min(cfg.a_floor, ev.A_base) - cfg.a_tol
    ok_floor = ev.A_cand >= floor
    _trace(rows, "contract_floor", ok_floor, A_base=round(ev.A_base, 4),
           A_cand=round(ev.A_cand, 4), floor=round(floor, 4))
    if not ok_floor:
        return Decision("REJECT", Reason.CONTRACT_FLOOR.value, rows,
                        {"A_base": ev.A_base, "A_cand": ev.A_cand, "floor": floor})

    # ---- 3. 判分稳定性（仅在启用 L2 时生效）----
    ok_judge = ev.judge_unstable_ratio <= cfg.judge_unstable_max
    _trace(rows, "judge_stable", ok_judge,
           unstable_ratio=ev.judge_unstable_ratio, max=cfg.judge_unstable_max)
    if not ok_judge:
        return Decision("REJECT", Reason.JUDGE_UNSTABLE.value, rows,
                        {"unstable_ratio": ev.judge_unstable_ratio})

    # ---- 3b. 靠削减计划提分（reward hacking）----
    # 只拦「计划变短 + A 上升」这个组合：单纯变短可能是真的变高效了。
    # 阈值用 0.5（一半以上任务出现该模式），避免个别任务的正常波动触发。
    ok_plan = ev.plan_undercut_rate < 0.5
    _trace(rows, "plan_not_undercut", ok_plan,
           undercut_rate=round(ev.plan_undercut_rate, 4))
    if not ok_plan:
        return Decision("REJECT", Reason.CONTRACT_FLOOR.value, rows,
                        {"plan_undercut_rate": ev.plan_undercut_rate,
                         "note": "A 的上升来自削减计划步数，而非执行质量提升"})

    # ---- 4. 局部崩溃（平均涨但个别任务崩）----    # 任务数 < 4 时「回归率」没有意义（1 个任务变差就是 100%），此时只看均值。
    if ev.n_tasks >= 4:
        ok_reg = ev.regression_rate <= cfg.r_max
        _trace(rows, "regression_rate", ok_reg, rate=ev.regression_rate,
               max=cfg.r_max, n_tasks=ev.n_tasks)
        if not ok_reg:
            return Decision("REJECT", Reason.REGRESSION.value, rows,
                            {"regression_rate": ev.regression_rate, "max": cfg.r_max})

    # ---- 5. 主目标：配对 t 的 δ ----
    pr = ev.paired(cfg.alpha)
    dQ, thr = pr.mean, pr.delta_crit
    _trace(rows, "primary_gain", dQ > thr, n=pr.n, mean=round(dQ, 4),
           delta_crit=round(thr, 4), p=round(pr.p_value, 6))

    if dQ < -thr:
        _trace(rows, "not_worse", False, mean=round(dQ, 4), delta_crit=round(thr, 4))
        return Decision("REJECT", Reason.REGRESSION.value, rows,
                        {"mean_delta": dQ, "delta_crit": thr, "n": pr.n})

    if dQ <= thr:
        # 未过噪声门的三条例外（ΔB 不在其中）
        if ev.dC < -cfg.c_save_min:
            _trace(rows, "exception_cheaper", True, dC=round(ev.dC, 4))
            return Decision("ACCEPT", Accept.CHEAPER_EQUAL.value, rows,
                            {"dC": ev.dC, "mean_delta": dQ})
        if ev.add_novel_component:
            _trace(rows, "exception_novelty", True)
            return Decision("ACCEPT", Accept.NOVELTY.value, rows, {"mean_delta": dQ})
        if ev.hard_failure_removed:
            _trace(rows, "exception_hard_failure", True)
            return Decision("ACCEPT", Accept.HARD_FAILURE_REMOVED.value, rows,
                            {"mean_delta": dQ})
        return Decision("REJECT", Reason.WITHIN_NOISE.value, rows,
                        {"mean_delta": dQ, "delta_crit": thr, "n": pr.n, "p": pr.p_value})

    # ---- 6. 成本规则：ΔC ≤ β0 + β1·ΔQ ----
    allowance = cfg.beta0 + cfg.beta1 * dQ
    ok_cost = ev.dC <= allowance
    _trace(rows, "cost_rule", ok_cost, dC=round(ev.dC, 4),
           allowance=round(allowance, 4), dQ=round(dQ, 4))
    if not ok_cost:
        return Decision("REJECT", Reason.COST_RULE.value, rows,
                        {"dC": ev.dC, "allowance": allowance, "dQ": dQ})

    return Decision("ACCEPT", Accept.VALIDATED.value, rows,
                    {"mean_delta": dQ, "delta_crit": thr, "n": pr.n, "dC": ev.dC})


# ---------------------------------------------------------------- 报告

def explain(decision: Decision) -> str:
    """给人类看的一行解释（CLI 与报告共用）。"""
    if decision.accepted:
        return f"ACCEPT({decision.reason})"
    first_fail = next((r for r in decision.gate_trace if not r["pass"]), None)
    if first_fail:
        return f"REJECT({decision.reason}) @ {first_fail['check']}"
    return f"REJECT({decision.reason})"
