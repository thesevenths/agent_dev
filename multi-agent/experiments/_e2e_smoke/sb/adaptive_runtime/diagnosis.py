"""诊断层：把 trace 映射成失败模式（taxonomy 必须被「可判定字段」约束）。

现状：历史日志里只有三类可判信号 —— ``kind=recursion / truncated / exception``
（95 条 run 共 36 次 step-failure，其中 recursion 28 次占 78%）。
因此首版只启用 5 类，其余一律 ``UNKNOWN`` 并统计占比：
    UNKNOWN > 40% → 停止加类别，先补埋点。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict, field

from .settings import CFG, Settings
from .trace_export import Trace


class Taxonomy:
    COMPLETION_FAILURE = "COMPLETION_FAILURE"   # 预算耗尽/截断，但产物已落盘
    EXECUTION_FAILURE = "EXECUTION_FAILURE"     # 预算耗尽/截断，且无产物
    REVIEW_FAILURE = "REVIEW_FAILURE"           # gate 反复拒绝
    PLANNING_FAILURE = "PLANNING_FAILURE"       # replan 空转
    TOOL_USAGE_FAILURE = "TOOL_USAGE_FAILURE"   # 工具异常
    UNKNOWN = "UNKNOWN"

    ALL = (COMPLETION_FAILURE, EXECUTION_FAILURE, REVIEW_FAILURE,
           PLANNING_FAILURE, TOOL_USAGE_FAILURE, UNKNOWN)


@dataclass
class Finding:
    taxonomy: str
    agent: str = ""
    step_no: int = 0
    reason: str = ""
    evidence: dict = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        return f"{self.taxonomy}|{self.agent or '-'}|{self.step_no}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fingerprint"] = self.fingerprint
        return d


def classify(trace: Trace, cfg: Settings | None = None) -> list[Finding]:
    cfg = cfg or CFG
    out: list[Finding] = []
    has_deliverable = len(trace.deliverables()) > 0

    for step in sorted(trace.steps.values(), key=lambda s: s.step):
        if step.failures == 0 and step.last != "reject":
            continue
        agent = step.agent or "-"
        if step.failures:
            kind = step.fail_kind
            if kind in ("recursion", "truncated"):
                # 有产物 → 只是没收尾；无产物 → 真失败
                tax = (Taxonomy.COMPLETION_FAILURE if has_deliverable
                       else Taxonomy.EXECUTION_FAILURE)
                why = next((e.get("why", "") for e in trace.events
                            if e.get("type") == "step_failure" and e.get("step_no") == step.step), "")
                out.append(Finding(tax, agent, step.step, reason=why[:200],
                                   evidence={"kind": kind, "attempts": step.failures,
                                             "has_deliverable": has_deliverable}))
            elif kind == "exception":
                why = next((e.get("why", "") for e in trace.events
                            if e.get("type") == "step_failure" and e.get("step_no") == step.step), "")
                tax = (Taxonomy.TOOL_USAGE_FAILURE if "tool" in why.lower()
                       else Taxonomy.EXECUTION_FAILURE)
                out.append(Finding(tax, agent, step.step, reason=why[:200],
                                   evidence={"kind": kind}))
            else:
                out.append(Finding(Taxonomy.UNKNOWN, agent, step.step,
                                   reason=f"unmapped kind={kind}", evidence={"kind": kind}))
        elif step.last == "reject" and step.rejects >= 1:
            out.append(Finding(Taxonomy.REVIEW_FAILURE, agent, step.step,
                               reason="gate 反复拒绝且未通过", evidence={"rejects": step.rejects}))

    replans = trace.counters.get("replan", 0)
    noop_max = 2  # AGENT_REPLAN_NOOP_MAX 默认值；该符号已冻结，不随候选变化
    if replans >= noop_max and trace.total_steps > 0:
        out.append(Finding(Taxonomy.PLANNING_FAILURE, "supervisor", 0,
                           reason=f"replan 触发 {replans} 次",
                           evidence={"replan": replans, "total_steps": trace.total_steps}))

    if not out:
        out.append(Finding(Taxonomy.UNKNOWN, "-", 0, reason="无失败信号"))
    return out


def summarize(findings: list[Finding], cfg: Settings | None = None) -> dict:
    cfg = cfg or CFG
    counts = Counter(f.taxonomy for f in findings)
    total = sum(counts.values()) or 1
    unknown_ratio = counts.get(Taxonomy.UNKNOWN, 0) / total
    hint = ""
    if unknown_ratio > cfg.unknown_max:
        hint = ("UNKNOWN 占比超过阈值：taxonomy 已超前于埋点，应先补字段而不是增加类别")
    return {
        "counts": dict(counts),
        "total": sum(counts.values()),
        "unknown_ratio": round(unknown_ratio, 4),
        "unknown_max": cfg.unknown_max,
        "hint": hint,
        "dominant": counts.most_common(1)[0][0] if counts else Taxonomy.UNKNOWN,
    }


def from_trace(trace: Trace, cfg: Settings | None = None) -> tuple[list[Finding], dict]:
    f = classify(trace, cfg)
    return f, summarize(f, cfg)
