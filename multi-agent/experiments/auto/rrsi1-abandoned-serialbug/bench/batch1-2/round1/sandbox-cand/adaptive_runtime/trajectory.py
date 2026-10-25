"""成功/失败轨迹对比（V3 §13.5 弱化版：跨任务分布对比）。

**必须诚实地标注边界**：这不是 ModularRSI 的严格定义（同一任务多次执行、逐步对照），
而是「跨任务把失败 run 与成功 run 的同步骤分布放一起看」。因此它只能说明
「失败 run 在 X 上显著更费」，**不能归因单条轨迹**。

正因为边界这么窄，它才值得做：只需要历史日志，零额外成本，却能回答
「CodeAgent 平均 41 次工具往返 vs 成功 run 18 次」这类问题 —— 而这往往直接指向
预算/停止条件，而不是提示词。
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .trace_export import Trace

#: 每条结论都必须带上这句，避免被当成因果结论引用
CAVEAT = ("弱化版：跨任务分布对比，不是同一任务的逐步对照 —— 只能说明「失败 run 在这些"
          "指标上更差」，不能归因到单条轨迹。")


def _is_ok(trace: Trace) -> bool:
    """成功 = 计划步骤全部通过且未截断/未兜底。"""
    return (trace.total_steps > 0 and trace.contract_rate >= 0.999
            and not trace.truncated and not trace.salvaged)


def split(traces: Iterable[Trace]) -> tuple[list[Trace], list[Trace]]:
    ok, bad = [], []
    for t in traces:
        (ok if _is_ok(t) else bad).append(t)
    return ok, bad


def cluster_failures(traces: Iterable[Trace]) -> list[dict]:
    """把失败按 ``(kind, agent, step_no)`` 聚类 —— 这是唯一有依据的分组维度。"""
    buckets: dict[tuple[str, str, int], int] = defaultdict(int)
    for t in traces:
        for no, st in t.steps.items():
            if st.last in ("fail", "unknown") or st.failures:
                buckets[(st.fail_kind or "unknown", st.agent or "unknown", no)] += 1
    rows = [{"kind": k, "agent": a, "step_no": n, "count": c}
            for (k, a, n), c in buckets.items()]
    rows.sort(key=lambda r: (-r["count"], r["kind"], r["agent"], r["step_no"]))
    return rows


def _tool_per_step(trace: Trace) -> float:
    tools = int(trace.counters.get("tool", 0))
    steps = max(1, trace.total_steps or len(trace.steps))
    return tools / steps


def _retry_per_step(trace: Trace) -> float:
    retries = sum(s.rejects for s in trace.steps.values())
    steps = max(1, trace.total_steps or len(trace.steps))
    return retries / steps


def _mean(xs: list[float]) -> float:
    return round(sum(xs) / len(xs), 3) if xs else 0.0


def _metric(ok: list[Trace], bad: list[Trace], name: str, fn) -> dict | None:
    a, b = [fn(t) for t in ok], [fn(t) for t in bad]
    if not a or not b:
        return None
    return {"metric": name, "ok_mean": _mean(a), "fail_mean": _mean(b),
            "diff": round(_mean(b) - _mean(a), 3), "n_ok": len(a), "n_fail": len(b)}


def contrast(traces: Iterable[Trace]) -> dict:
    """输出成功/失败两组在若干可计算指标上的分布对比。"""
    ok, bad = split(traces)
    metrics = [
        _metric(ok, bad, "工具往返 / 步", _tool_per_step),
        _metric(ok, bad, "重试次数 / 步", _retry_per_step),
        _metric(ok, bad, "契约达成率", lambda t: t.contract_rate),
        _metric(ok, bad, "通过步骤数", lambda t: float(t.passed_steps)),
    ]
    return {"n_ok": len(ok), "n_fail": len(bad),
            "metrics": [m for m in metrics if m],
            "failure_clusters": cluster_failures(bad),
            "caveat": CAVEAT,
            "usable": bool(ok) and bool(bad)}


def by_agent(traces: Iterable[Trace]) -> list[dict]:
    """按 agent 统计「失败 run 与成功 run 的每步重试数」。agent 名来自 trace 的 step 状态。"""
    ok, bad = split(traces)
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"ok": [], "fail": []})
    for group, items in (("ok", ok), ("fail", bad)):
        for t in items:
            steps = max(1, t.total_steps or len(t.steps))
            per: dict[str, int] = defaultdict(int)
            for st in t.steps.values():
                if st.agent:
                    per[st.agent] += st.rejects
            for agent, n in per.items():
                acc[agent][group].append(n / steps)
    rows = []
    for agent, d in acc.items():
        if d["ok"] and d["fail"]:
            rows.append({"agent": agent,
                         "ok_retry_per_step": _mean(d["ok"]),
                         "fail_retry_per_step": _mean(d["fail"]),
                         "diff": round(_mean(d["fail"]) - _mean(d["ok"]), 3),
                         "n_ok": len(d["ok"]), "n_fail": len(d["fail"])})
    rows.sort(key=lambda r: -r["diff"])
    return rows


def to_markdown(report: dict) -> str:
    lines = ["# 成功 / 失败轨迹对比（弱化版）", "",
             f"> {report['caveat']}", "",
             f"- 成功 run：{report['n_ok']} / 失败 run：{report['n_fail']}", ""]
    if not report["usable"]:
        lines += ["**样本不足**：成功与失败两组都需要至少 1 个 run 才能对比。", ""]
    if report["metrics"]:
        lines += ["## 分布对比", "", "| 指标 | 成功 run 均值 | 失败 run 均值 | 差 |",
                  "|---|---|---|---|"]
        for m in report["metrics"]:
            lines.append(f"| {m['metric']} | {m['ok_mean']} | {m['fail_mean']} | {m['diff']:+.3f} |")
        lines.append("")
    if report.get("by_agent"):
        lines += ["## 按 agent 的每步重试数", "", "| agent | 成功 | 失败 | 差 |", "|---|---|---|---|"]
        for r in report["by_agent"]:
            lines.append(f"| {r['agent']} | {r['ok_retry_per_step']} | "
                         f"{r['fail_retry_per_step']} | {r['diff']:+.3f} |")
        lines.append("")
    if report["failure_clusters"]:
        lines += ["## 失败聚类（kind × agent × step）", "", "| kind | agent | step | 次数 |",
                  "|---|---|---|---|"]
        for c in report["failure_clusters"][:15]:
            lines.append(f"| {c['kind']} | {c['agent']} | {c['step_no']} | {c['count']} |")
        lines.append("")
    return "\n".join(lines)
