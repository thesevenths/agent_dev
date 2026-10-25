"""诊断层：把 trace 映射成失败模式（taxonomy 必须被「可判定字段」约束）。

设计约束
--------
taxonomy 的每一类都必须能被**确定性字段**判定，不允许 LLM 自评。
原因：如果「对不对」由模型自己说，那么闭环会退化成自我证成——
模型学会的不是把任务做对，而是把自我描述写得像做对了。

两条信号通道
------------
1. **硬失败通道**（run 没跑完）：``kind=recursion / truncated / exception``
   —— 来自 ``[step-failure]`` 行，以及 gate 反复 reject / replan 空转。
   历史基线：95 条 run 共 36 次 step-failure，其中 recursion 28 次占 78%。
2. **质量缺陷通道**（run 跑完了，但产出不达标）：证据率、完整性。
   这条通道是必需的，因为最大的失分项往往不在「崩没崩」——
   实测一组成熟 run 的 A/C 已接近满分，27 分差距全部落在 E
   （268 个关键数字仅 27 个带来源）。只看硬失败，闭环永远发现不了它。

其余一律 ``UNKNOWN`` 并统计占比：UNKNOWN > 40% → 停止加类别，先补埋点。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict, field
from typing import TYPE_CHECKING

from .settings import CFG, Settings
from .trace_export import Trace

if TYPE_CHECKING:  # 仅用于类型标注，避免与 evaluator 形成运行期依赖
    from .evaluator import Score


class Taxonomy:
    # ---- 硬失败通道：执行没走完 ----
    COMPLETION_FAILURE = "COMPLETION_FAILURE"   # 预算耗尽/截断，但产物已落盘
    EXECUTION_FAILURE = "EXECUTION_FAILURE"     # 预算耗尽/截断，且无产物
    REVIEW_FAILURE = "REVIEW_FAILURE"           # gate 反复拒绝
    PLANNING_FAILURE = "PLANNING_FAILURE"       # replan 空转
    TOOL_USAGE_FAILURE = "TOOL_USAGE_FAILURE"   # 工具异常

    # ---- 质量缺陷通道：执行走完了，但产出不达标 ----
    #: 产物中的量化断言缺来源。跨项目通用：任何产出「报告 / 分析 / 结论」的
    #: 任务都受此约束，与具体业务域无关。
    EVIDENCE_FAILURE = "EVIDENCE_FAILURE"
    #: 计划产物缺失、截断或兜底，导致 C 低于阈值。
    COMPLETENESS_FAILURE = "COMPLETENESS_FAILURE"

    UNKNOWN = "UNKNOWN"

    HARD = (COMPLETION_FAILURE, EXECUTION_FAILURE, REVIEW_FAILURE,
            PLANNING_FAILURE, TOOL_USAGE_FAILURE)
    QUALITY = (EVIDENCE_FAILURE, COMPLETENESS_FAILURE)

    ALL = HARD + QUALITY + (UNKNOWN,)


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


def classify(trace: Trace, cfg: Settings | None = None,
             score: "Score | None" = None) -> list[Finding]:
    """把一次 run 的 trace（可选地带上评分）映射为失败模式列表。

    ``score`` 传入时启用质量缺陷通道。不传则行为与首版一致（仅硬失败），
    以便历史产物可被复现。
    """
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

    if score is not None:
        out.extend(quality_findings(trace, score, cfg))

    if not out:
        out.append(Finding(Taxonomy.UNKNOWN, "-", 0, reason="无失败信号"))
    return out


def quality_findings(trace: Trace, score: "Score", cfg: Settings) -> list[Finding]:
    """质量缺陷通道：从评分里反推「这次跑完了，但哪里不达标」。

    与硬失败通道的关键区别：这里**没有**崩栈、没有异常、每一步 gate 都 PASS。
    若只看 ``step.failures``，这类 run 会被判成「无失败信号」，
    于是闭环认为无改进空间 —— 而这恰恰是 Q 的最大失分区。

    判定全部走确定性字段，证据里附上「未带来源的关键数字」样本，
    保证每条 Finding 都能追溯到成品文本的具体位置。
    """
    out: list[Finding] = []
    detail = score.detail or {}

    ed = detail.get("evidence") or {}
    claims = int(ed.get("claims") or 0)
    supported = int(ed.get("supported") or 0)

    if (score.E is not None and claims >= cfg.evidence_min_claims
            and score.E < cfg.evidence_floor):
        out.append(Finding(
            Taxonomy.EVIDENCE_FAILURE,
            agent=_last_producing_agent(trace),
            step_no=0,
            reason=(f"证据率 {score.E:.3f} 低于阈值 {cfg.evidence_floor:.2f}"
                    f"（{supported}/{claims} 个量化断言带来源）"),
            evidence={
                "claims": claims,
                "supported": supported,
                "unsupported_samples": list(ed.get("unsupported_samples") or [])[:3],
            },
        ))

    if score.C is not None and score.C < cfg.completeness_floor:
        cdet = detail.get("completeness") or {}
        out.append(Finding(
            Taxonomy.COMPLETENESS_FAILURE,
            agent="-",
            step_no=0,
            reason=f"完整性 {score.C:.3f} 低于阈值 {cfg.completeness_floor:.2f}",
            evidence={
                "planned": score.planned,
                "truncated": bool(cdet.get("truncated")),
                "salvaged": bool(cdet.get("salvaged")),
                "missing": cdet.get("missing", 0),
            },
        ))

    return out


def _last_producing_agent(trace: Trace) -> str:
    """产出最终交付物的那一步由谁执行 —— 用于把质量缺陷归到具体 agent。

    归不到就返回 ``-``，让 fingerprint 走 UNKNOWN 聚合，不猜测归属。
    """
    agents = [s.agent for s in sorted(trace.steps.values(), key=lambda s: s.step)
              if s.agent]
    return agents[-1] if agents else "-"


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


def from_trace(trace: Trace, cfg: Settings | None = None,
               score: "Score | None" = None) -> tuple[list[Finding], dict]:
    f = classify(trace, cfg, score=score)
    return f, summarize(f, cfg)


def from_score(score: "Score", cfg: Settings | None = None) -> list[Finding]:
    """仅从评分产出质量缺陷（无 trace 场景，例如 replay 模式的批量复盘）。"""
    cfg = cfg or CFG
    return quality_findings(Trace(run_id=score.run_id), score, cfg)
