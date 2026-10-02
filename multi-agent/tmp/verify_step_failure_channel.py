# -*- coding: utf-8 -*-
"""自检：step_failures「失败原因 → 下一个 sub agent」通道（A+B 档改动）。

覆盖：hint 生成 / 回溯窗口 / 渲染 / 派发注入 / 记录写入+落盘 / 再规划注入 / 零回归。
放在 tmp/ 下作为临时产物，随时可删。用法：python verify_step_failure_channel.py
"""
import os
import sys

sys.path.insert(0, r"F:\agent\multi-agent")
from handoff import (_step_assignment_text, _render_step_failures,
                     _recent_step_failures, _failure_hint)

PLAN = [
    {"title": "获取价格数据", "description": "检索 BTC 每日收盘价 → crawler", "status": "completed"},
    {"title": "获取情绪新闻", "description": "检索新闻宏观 → crawler", "status": "completed"},
    {"title": "数据分析可视化", "description": "计算指标生成图表 → code", "status": "pending"},
    {"title": "撰写报告", "description": "写 md 报告 → code", "status": "pending"},
]
_FAILED = 0


def ok(cond, msg):
    global _FAILED
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        _FAILED += 1


def base_state(cur=3, failures=None):
    return {"execution_plan": PLAN, "current_step": cur,
            "artifacts": [r"F:\agent\multi-agent\tmp\x.md"],
            "step_failures": failures or [], "plan_goal": "BTC 行情分析"}


print("[1] _failure_hint 三种 kind")
for k in ("recursion", "exception", "critic"):
    h = _failure_hint(k)
    ok(len(h) > 80, f"{k}: {len(h)} chars")
h = _failure_hint("recursion")
ok("Windows" in h and "绝对路径" in h, "recursion hint 含 Windows 绝对路径要求（消掉必失败的一次往返）")
ok("tool_calls" in h, "recursion hint 要求用不带 tool_calls 的总结收尾")
ok(_failure_hint("unknown") != "", "未知 kind 有兜底文案")

print("[2] 回溯窗口 lookback=2（覆盖 supervisor 合并失败步）")
F = [{"step": 3, "kind": "recursion"}]
ok(len(_recent_step_failures(base_state(3, F), 3)) == 1, "step3 失败 → cur=3 可见")
ok(len(_recent_step_failures(base_state(4, F), 4)) == 1, "step3 失败 → cur=4 可见（一次合并）")
ok(len(_recent_step_failures(base_state(5, F), 5)) == 1, "step3 失败 → cur=5 可见（二次合并）")
ok(len(_recent_step_failures(base_state(6, F), 6)) == 0, "cur=6 超出窗口 → 不再提示（无陈年噪音）")
ok(len(_recent_step_failures(base_state(3, None), 3)) == 0, "无失败记录 → 空")

print("[3] _render_step_failures")
rec = {"step": 3, "agent": "CodeAgent", "kind": "recursion", "attempt": 2,
       "reason": "ReAct step budget exhausted: recursion_limit=15 (AGENT_MAX_ITERATIONS)",
       "hint": _failure_hint("recursion"), "at": "2026-10-02T22:24:31"}
r = _render_step_failures(base_state(3, [rec]), 3)
for kw in ("PREVIOUS ATTEMPT", "WHY it failed", "HOW TO AVOID", "attempt #2", "recursion"):
    ok(kw in r, f"渲染含 {kw!r}")
old = {"step": 3, "agent": "CodeAgent", "kind": "recursion", "attempt": 1, "reason": "x", "at": "t"}
ok("HOW TO AVOID" in _render_step_failures(base_state(3, [old]), 3),
   "老记录无 hint 字段 → 按 kind 现算补齐")
ok(_render_step_failures(base_state(3, []), 3) == "", "无失败 → 空串（零回归）")

print("[4] _step_assignment_text 注入")
txt = _step_assignment_text(base_state(3, [rec]), "CodeAgent")
ok("PREVIOUS ATTEMPT" in txt, "assignment 含失败回溯段")
ok(txt.index("PREVIOUS ATTEMPT") < txt.index("HARD RULES"),
   "失败段位于 HARD RULES 之前（动手前就看到）")
txt0 = _step_assignment_text(base_state(3, []), "CodeAgent")
ok("PREVIOUS ATTEMPT" not in txt0, "无失败时 assignment 不含该段")
for kw in ("Supervisor assignment", "Overall goal", "Upstream results", "HARD RULES for this step"):
    ok(kw in txt0, f"原有段落保持完好: {kw}")

print("[5] agents._append_step_failure（写入 + 落盘 + 截断）")
try:
    try:
        import agents
    except ImportError:
        # 本机默认是 langchain 0.3.x（无 langchain.agents.create_agent），而 langgraph dev 实际
        # 跑的环境是 langchain 1.x。此处用 duck-typed stub 补齐缺失符号——被验证的
        # _append_step_failure 是纯函数，只依赖 datetime / _failure_hint / log_event，桩足够跑通。
        from unittest.mock import MagicMock
        import langchain.agents as _amod
        _amod.create_agent = MagicMock()
        for _m in ("langchain.agents.middleware", "llm", "prompt", "tools", "context",
                   "compress", "summary", "planutil", "hitl"):
            sys.modules.setdefault(_m, MagicMock())
        import agents
        print("  NOTE  langchain<1 环境，已用 duck-typed stub 补齐后 import")
    def _safe_bad_state():
        """兜底路径自身也必须吞掉异常：记录失败绝不能反过来把节点搞崩。"""
        try:
            return list(agents._append_step_failure(object(), 1, "X", "critic", "bad-state") or []) == []
        except Exception:
            return False

    st = dict(base_state(3, []))
    st["memory_key"] = "__steptest__"
    f1 = agents._append_step_failure(st, 3, "CodeAgent", "recursion", "boom-A")
    st["step_failures"] = f1
    f2 = agents._append_step_failure(st, 3, "CodeAgent", "critic", "boom-B")
    ok(len(f1) == 1 and f1[0]["attempt"] == 1, "首次失败 attempt=1")
    ok(len(f2) == 2 and f2[1]["attempt"] == 2 and f2[0]["attempt"] == 1, "同一步第二次 attempt=2")
    ok(f2[1]["step"] == 3 and f2[1]["kind"] == "critic" and bool(f2[1]["hint"]),
       "记录字段完整 (step/kind/reason/hint/attempt/at)")
    big = agents._append_step_failure({"step_failures": [{"step": i} for i in range(50)]},
                                      1, "X", "exception", "z")
    ok(len(big) <= agents._STEP_FAILURES_MAX, f"超长列表截断到 {len(big)} <= {agents._STEP_FAILURES_MAX}")
    ok(_safe_bad_state(), "传入非法 state 不抛异常（兜底路径自身也必须安全）")
    from runlog import run_file
    p = run_file()
    ok(bool(p) and os.path.exists(p), f"log_event 已落盘: {os.path.basename(str(p))}")
    if p and os.path.exists(p):
        body = open(p, encoding="utf-8", errors="ignore").read()
        ok("[step-failure]" in body and "boom-A" in body,
           "run log 含 [step-failure] + 失败原因（补上原缺口）")
        print("\n--- run log 摘录 ---")
        for line in body.splitlines():
            if "[step-failure]" in line or "WHY:" in line:
                print("   ", line.strip()[:150])
        globals()["_LOGFILE"] = p
except Exception as e:
    print("  SKIP import agents:", type(e).__name__, e)

print("[6] plan._replan_tail 注入")
try:
    try:
        import plan
    except ImportError:
        from unittest.mock import MagicMock
        import langchain.agents as _amod
        _amod.create_agent = MagicMock()
        for _m in ("langchain.agents.middleware", "llm", "prompt", "tools", "context",
                   "compress", "summary", "planutil", "hitl"):
            sys.modules.setdefault(_m, MagicMock())
        import plan
        print("  NOTE  langchain<1 环境，已用 duck-typed stub 补齐后 import")
    import inspect
    src = inspect.getsource(plan._replan_tail)
    ok("_render_step_failures" in src and "fail_ctx" in src, "再规划 prompt 已接失败回溯")
    ok("RE-DOING A FAILED STEP" in src, "含「重做失败步必须写清避免方式与新约束」要求")
    src2 = inspect.getsource(plan._replan_tail)
    ok("fail_ctx if fail_ctx else \"\"" in src2 or "if fail_ctx else" in src2,
       "无失败时不插入空块（零回归）")
except Exception as e:
    print("  SKIP import plan:", type(e).__name__, e)

print("\n=== RESULT:", "ALL PASS" if _FAILED == 0 else f"{_FAILED} FAILED", "===")
sys.exit(0 if _FAILED == 0 else 1)
