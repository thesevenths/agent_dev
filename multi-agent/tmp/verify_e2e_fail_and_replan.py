# -*- coding: utf-8 -*-
"""端到端闭环验证：sub agent 失败判定 + supervisor 计划更新。

与已有脚本的区别：
  verify_replan_evidence.py      → 只打 _should_replan（证据门）
  verify_replan_and_gates.py     → 只打各判定函数（单元）
  verify_budget_and_noop.py      → 预算与空转
  本脚本                          → 真实调用 supervisor() 节点函数，走完整
                                    「失败 state → 证据门 → LLM(prompt) → 应用 plan → 路由 next」
                                    闭环，并验证两个此前从未验证的结构性边界。

运行：PYTHONIOENCODING=utf-8 python tmp/verify_e2e_fail_and_replan.py
"""
import os
import sys
import json
import time
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 探针日志隔离到独立 memory_key，结束后清理
MK = "__e2e_probe__"
TMP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tmp")

PASS = FAIL = 0


def ok(cond, name, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}" + (f"\n          -> {detail}" if detail else ""))


def section(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


# ---------------------------------------------------------------- stub 依赖
# 本机默认 python 的 langchain 版本/缺失依赖与项目实际运行环境（langgraph dev）不同，
# 用「缺什么 stub 什么」的循环把重依赖挡掉，只验证纯逻辑分支。
from unittest.mock import MagicMock  # noqa: E402

import langchain.agents as _amod  # noqa: E402
if not hasattr(_amod, "create_agent"):
    _amod.create_agent = MagicMock()

for _ in range(15):
    try:
        import plan  # noqa: E402
        import agents  # noqa: E402
        import supervisor as sup  # noqa: E402
        break
    except ModuleNotFoundError as e:
        n = e.name or ""
        if not n:
            raise
        sys.modules[n] = MagicMock()
        if "." in n:
            p, c = n.rsplit(".", 1)
            if isinstance(sys.modules.get(p), MagicMock):
                setattr(sys.modules[p], c, sys.modules[n])
    except ImportError as e:
        import re as _re
        m = _re.search(r"cannot import name '([^']+)' from '([^']+)'", str(e))
        if not m:
            raise
        mod = sys.modules.get(m.group(2))
        if isinstance(mod, MagicMock):
            continue
        if mod is not None:
            setattr(mod, m.group(1), MagicMock())
        else:
            sys.modules[m.group(2)] = MagicMock()

if not isinstance(getattr(plan, "_structured_with_retry", None), MagicMock):
    try:
        plan._structured_with_retry  # noqa: B018
    except AttributeError:
        plan._structured_with_retry = MagicMock()

# 捕获喂给 LLM 的 prompt，并可控地返回再规划结果
CAPTURED = {}
REPLY = {}


def _fake_structured(llm, messages, schema, label="", **kw):
    CAPTURED["messages"] = messages
    CAPTURED["label"] = label
    return REPLY.get("parsed")


plan._structured_with_retry = _fake_structured
sup._summarize_observations = lambda state: "· 已完成：抓取价格数据、抓取情绪新闻"
sup._extract_longterm = lambda state, key: None

# ---------------------------------------------------------------- 数据
def step(title, desc, agent, status="pending"):
    return {"title": title, "description": f"{desc} → {agent}", "status": status}


PLAN5 = [
    step("抓取价格", "抓取 BTC 日线", "crawler_agent", "completed"),
    step("抓取情绪", "抓取新闻与情绪", "crawler_agent", "completed"),
    step("分析可视化", "计算指标生成图表", "code_agent", "pending"),
    step("撰写报告", "撰写 Markdown 报告", "code_agent", "pending"),
    step("发送邮件", "将报告通过邮件发送给用户", "chat_agent", "pending"),
]


def base_state(plan, current, failures=None, artifacts=None, obs=None):
    return {
        "messages": [],
        "observations": obs or [],
        "execution_plan": [dict(s) for s in plan],
        "current_step": current,
        "artifacts": artifacts or [],
        "step_failures": failures or [],
        "plan_goal": "分析比特币近期行情并发送报告",
        "memory_key": MK,
        "run_started_at": time.time() - 60,
        "replan_noop_streak": 0,
        "error_count": 0,
    }


def run_sup(state, reply):
    REPLY["parsed"] = reply
    CAPTURED.clear()
    return sup.supervisor(state)


# ================================================================ Q2 闭环
section("Q2-A：失败后 supervisor 是否真的触发再规划（不跳过）")

st = base_state(PLAN5, 3, failures=[{
    "step": 3, "agent": "CodeAgent", "kind": "recursion",
    "reason": "ReAct step budget 28 exhausted → TRUNCATED",
    "hint": "脚本跑通后立刻收尾，禁止再打磨", "attempt": 1, "at": "2026-10-02T23:46",
}])
# 真实链路：子 agent 失败后由 _mark_failed 把 index=cur-1 那步标 failed
st["execution_plan"] = plan._mark_failed(st["execution_plan"], 2)
res = run_sup(st, {"next": "code_agent", "reason": "merge redo",
                   "execution_plan": [
                       dict(PLAN5[0]), dict(PLAN5[1]),
                       {"title": "重做分析并出报告", "description": "一次性完成分析与报告 → code_agent",
                        "status": "pending"},
                       dict(PLAN5[4]),
                   ]})
ok(res["next"] == "code_agent", "有 failed 步 → 再规划被触发，next=code_agent")
ok(CAPTURED.get("label") == "re-plan",
   "确实调用了再规划 LLM（label=re-plan），未走 SKIPPED",
   detail=f"CAPTURED keys={list(CAPTURED.keys())}")

section("Q2-B：失败原因是否真的进了再规划 prompt（supervisor 能看到根因吗）")

user_msg = "\n".join(
    str(getattr(m, "content", "")) for m in (CAPTURED.get("messages") or [])
)
ok("recursion" in user_msg.lower(), "失败 kind=recursion 出现在 prompt 中")
ok("truncated" in user_msg.lower() or "exhausted" in user_msg.lower(),
   "失败根因文本（TRUNCATED/exhausted）进入 prompt",
   detail=user_msg[:400])
ok("how to avoid" in user_msg.lower() or "hint" in user_msg.lower(),
   "改进办法（hint / HOW TO AVOID）进入 prompt")

section("Q2-C【关键】失败步能否被重新派发（重做同一索引的步）")

# 模型想把 failed 的 step3 原样重做：返回 next=code_agent 且不改计划。
st2 = base_state(PLAN5, 3, failures=[{"step": 3, "agent": "CodeAgent", "kind": "recursion",
                                      "reason": "budget exhausted", "hint": "h", "attempt": 1,
                                      "at": "t"}])
res2 = run_sup(st2, {"next": "code_agent", "reason": "redo step 3"})
p2 = res2["execution_plan"]
ok(p2[2].get("status") != "pending",
   f"failed 步不可能回到 pending 重做：step3 状态被推进为 {p2[2].get('status')!r}",
   detail="current_step 单调递增，_replan_tail 明确不回退；failed 步 index<current 永不再派发")
ok(res2["next"] != "code_agent" or p2[2].get("status") != "pending",
   "重做只能靠把失败内容合并进后续 pending 步（模型需自行改写 description）")

section("Q2-D【关键】最后一步失败会被 FINISH 直接吞掉吗")

st3 = base_state(PLAN5, 5, failures=[{"step": 5, "agent": "ChatAgent", "kind": "exception",
                                      "reason": "邮件发送失败", "hint": "h", "attempt": 3,
                                      "at": "t"}])
PLAN5_LAST_FAILED = [dict(s) for s in PLAN5]
PLAN5_LAST_FAILED[4]["status"] = "failed"
st3["execution_plan"] = PLAN5_LAST_FAILED
res3 = run_sup(st3, {"next": "FINISH", "reason": "done"})
ok(res3.get("next") == "FINISH",
   f"最后一步 failed → current(5)>=len(plan)(5) 直接走 FINISH 分支，next={res3.get('next')}")
ok(res3["execution_plan"][4].get("status") == "failed",
   "plan 终态确实保留了 failed 标记（没被洗成 completed）")
ok(CAPTURED.get("label") is None,
   "❗但 supervisor **完全没有调用 LLM**：失败步既未重试、也未改写、未告知用户，仅落一行日志",
   detail="supervisor.py:49 `if current >= len(plan)` 先于再规划判定，直接 FINISH")

section("Q2-E【关键】plan 长度只减不增 → 无剩余步时没有重做载体")

# 只剩最后一步且它 failed，模型想插入一个新的重试步
st4 = base_state(PLAN5, 4, failures=[{"step": 4, "agent": "CodeAgent", "kind": "critic",
                                      "reason": "报告质量不合格", "hint": "h", "attempt": 2,
                                      "at": "t"}])
want = [dict(PLAN5[0]), dict(PLAN5[1]), dict(PLAN5[2]),
        {"title": "重试分析", "description": "再算一次指标 → code_agent", "status": "pending"},
        {"title": "重试报告", "description": "再写一次报告 → code_agent", "status": "pending"},
        dict(PLAN5[4])]
res4 = run_sup(st4, {"next": "code_agent", "reason": "insert retry step", "execution_plan": want})
p4 = res4["execution_plan"]
ok(len(p4) == len(PLAN5),
   f"模型想插入重试步（{len(want)} 步）→ 被截断回原长 {len(p4)} 步（只减不增）",
   detail="plan.py:339 `if len(new_plan) > len(plan): new_plan = new_plan[:len(plan)]`")

section("Q2-F：证据门与空转停用的优先级（证据是否能压过停用）")

st5 = base_state(PLAN5, 3, failures=[])
st5["replan_noop_streak"] = 5
bad = os.path.join(TMP, "_probe_e2e_bad.json")
with open(bad, "w", encoding="utf-8") as f:
    f.write("{not json")
st5["artifacts"] = [bad]
should, why = plan._should_replan(st5, [dict(s) for s in PLAN5], 3)
ok(should, f"空转 streak=5 但有产物契约证据 → 仍触发再规划：{why[:60]}")

os.remove(bad)

# ================================================================ Q1 判定
section("Q1-A：failed 状态不会被后续推进洗成 completed")

pl = [dict(s) for s in PLAN5]
pl[2]["status"] = "failed"
after = plan._mark_progress(pl, 4)
ok(after[2].get("status") == "failed", "_mark_progress 不覆盖 failed（保真）")
ok(after[0].get("status") == "completed" and after[3].get("status") == "completed",
   "其余已完成步正常标 completed")

section("Q1-B：截断挽救语义（截断≠失败）")

import agents  # noqa: E402

fresh = os.path.join(TMP, "_probe_e2e_fresh.md")
with open(fresh, "w", encoding="utf-8") as f:
    f.write("# 比特币行情分析报告\n正文内容")
hits = agents._salvage_truncated_artifacts(time.time() - 30, [])
ok(fresh in hits, f"本步期间落盘的产物被识别（{len(hits)} 个）→ 走 critic 而非判死 failed")
ok(agents._salvage_truncated_artifacts(time.time() - 30, [fresh]) == [],
   "已在 artifacts 清单里的产物被排除（不重复挽救）")
ok(agents._salvage_truncated_artifacts(time.time() + 3600, []) == [],
   "时间窗之外（起点在未来）→ 不误把上游文件当本步产物（防错误挽救）")
os.remove(fresh)

section("Q1-C：动作契约（要求的动作是否真的发生） —— 返回 (checked, passed, reason)")

act = agents._action_contract_gate
gate_r = act("将最终报告通过邮件发送给用户 → chat_agent", "已读取报告内容", "read_file")
ok(gate_r[0] is True, "步骤要求发邮件 → 动作契约生效（checked=True）")
ok(gate_r[1] is False,
   f"只 read_file 却要求发邮件 → 判不合格：{str(gate_r[2])[:70]}")
gate_p = act("将最终报告通过邮件发送给用户 → chat_agent", "邮件已发送", "send_email")
ok(gate_p[0] is True and gate_p[1] is True, "有 send_email 工具痕迹 → 放行")
gate_n = act("计算 BTC 的 MA5 与 RSI 指标 → code_agent", "算完了", "shell_exec")
ok(gate_n[0] is False and gate_n[1] is True,
   "纯分析步不触发动作契约（checked=False，无 false positive）")

# ---------------------------------------------------------------- 清理
print("\n" + "-" * 78)
for d in ("", "log"):
    pass
logdir = os.path.join(os.path.dirname(TMP), "log")
removed = 0
if os.path.isdir(logdir):
    for fn in os.listdir(logdir):
        if MK in fn:
            try:
                os.remove(os.path.join(logdir, fn))
                removed += 1
            except OSError:
                pass
print(f"探针日志清理：{removed} 个")

print("-" * 78)
print(f"PASS={PASS}  FAIL={FAIL}")
print("\n说明：FAIL 项若为「结构性缺口」，代表当前设计如此（需改动语义才能修），非代码 bug。")
sys.exit(1 if FAIL else 0)
