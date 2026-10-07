"""验证 supervisor 的“同线程接着聊”修复（mock LLM，确定性、不依赖外部网络）。

判定改用「提交序号」：run_start 每次新提交生成 _submission_id；计划创建时记入 _plan_submission_id；
supervisor 在「计划已完成 且 _submission_id != _plan_submission_id」时判定为续问。
这与 messages 末尾是否被 langgraph dev 注入“对话摘要”完全无关（修复 2026-10-07 线上 FINISH 实证）。

核心断言：
  A. 计划已完成 + 新提交(_submission_id≠_plan_submission_id)  -> 必须重新规划（不 FINISH）。
  B. 计划已完成 + 同提交(_submission_id==_plan_submission_id) -> 维持 FINISH（向后兼容单次任务）。
  C. LIGHT 轻量微调追问 -> 路由上一轮 agent + 单步计划，不重规划。
  D. 【真实 bug 复现】messages[-1] 是平台注入的“对话摘要”合成消息，但提交序号不同
     -> 仍必须重新规划（不能因 messages[-1] 非干净用户消息而误 FINISH）。
"""
import sys
sys.path.insert(0, r"F:\agent\multi-agent")

from langchain_core.messages import HumanMessage, AIMessage
import supervisor as S
from state import PlanStep


# --- mock：让 情况2 的 _structured_with_retry 直接返回一份合法 plan，避免真实调 LLM ---
def _fake_structured(llm, messages, schema, label=None):
    return {
        "goal": "（追问）对比标普500",
        "execution_plan": [
            {"title": "获取标普500数据", "description": "搜索 SPX 资金流向", "status": "pending"},
            {"title": "对比分析", "description": "与纳指对比", "status": "pending"},
        ],
    }


S._structured_with_retry = _fake_structured

_PLANNED = "SUB_PLAN"   # 计划创建时的提交序号


def make_state(sub_diff=True, tail=None):
    """构造“首轮已结束”的线程状态。

    sub_diff=True  -> 本次提交序号与计划创建时不同（= 续问）
    tail           -> 追加在末尾的消息列表（用于模拟平台注入摘要 / 真实追问位置）
    """
    msgs = [HumanMessage(content="分析纳斯达克100指数近3年资金流向")]
    msgs.append(AIMessage(content="（上一轮的最终报告正文）"))
    if tail is not None:
        msgs.extend(tail)
    return {
        "memory_key": "test_thread",
        "messages": msgs,
        "execution_plan": [
            PlanStep(title="获取纳指数据", description="x", status="completed"),
            PlanStep(title="撰写报告", description="y", status="completed"),
        ],
        "current_step": 2,
        "plan_goal": "分析纳斯达克100",
        "run_started_at": None,
        "recalled_memory": "",
        "plan_summary": "",
        "observations": [],
        "summary_obs_seen": 0,
        "replan_noop_streak": 0,
        "plan_grown": 0,
        "step_failures": [],
        "error_count": 0,
        "snapshot_id": None,
        "sender": None,
        "next": None,
        "reason": None,
        "hallucination_check": None,
        "artifacts": [],
        "obs_total": 0,
        "final_note": None,
        "_submission_id": "SUB_NEW" if sub_diff else _PLANNED,
        "_plan_submission_id": _PLANNED,
    }


print("=== 单测 _is_new_user_followup / _starts_with_inject_prefix ===")
assert S._is_new_user_followup(HumanMessage(content="那标普500呢？")) is True
assert S._is_new_user_followup(HumanMessage(content="[System context] Today... 分析纳指")) is False
assert S._is_new_user_followup(HumanMessage(content="[Long-term memory about the USER...] x")) is False
assert S._is_new_user_followup(HumanMessage(content="Here is a summary of the conversation to date: ...")) is False
assert S._is_new_user_followup(AIMessage(content="报告")) is False
assert S._is_new_user_followup(None) is False
print("  ok: 追问判定正确（raw 用户=True；[System context]/[Long-term memory]/平台摘要=False；AIMessage=False）")


print("\n=== 用例 A：新提交(续问) -> 应重新规划（不 FINISH）===")
out_a = S.supervisor(make_state(sub_diff=True,
                                tail=[HumanMessage(content="那顺带对比一下标普500呢？")]))
nxt_a, plan_a = out_a.get("next"), out_a.get("execution_plan") or []
print("  next =", nxt_a, "| plan 步数 =", len(plan_a), "| current_step =", out_a.get("current_step"))
assert nxt_a and nxt_a != "FINISH", f"BUG: 续问却 FINISH 了 (next={nxt_a})"
assert len(plan_a) > 0 and out_a.get("current_step") == 1
print("  ok: 同线程续问（提交序号不同）被正确识别并重新规划")


print("\n=== 用例 B：同提交(无续问) -> 维持 FINISH ===")
out_b = S.supervisor(make_state(sub_diff=False))
nxt_b = out_b.get("next")
print("  next =", nxt_b)
assert nxt_b == "FINISH", f"BUG: 无续问却没 FINISH (next={nxt_b})"
print("  ok: 同一次提交、计划已完成 -> 维持原 FINISH（向后兼容）")


print("\n=== 用例 D：【真实 bug 复现】messages[-1]=平台注入摘要，但提交序号不同 -> 仍须重规划 ===")
# 模拟 langgraph dev 线程恢复：在末尾注入“对话摘要”，用户真实追问在其之前（独立 HumanMessage）
tail_injected = [
    HumanMessage(content="那顺带对比一下标普500呢？"),         # 真实追问（在摘要之前）
    HumanMessage(content="Here is a summary of the conversation to date:\n## SESSION I\n..."),  # 平台注入
]
out_d = S.supervisor(make_state(sub_diff=True, tail=tail_injected))
nxt_d, plan_d = out_d.get("next"), out_d.get("execution_plan") or []
print("  next =", nxt_d, "| plan 步数 =", len(plan_d))
assert nxt_d and nxt_d != "FINISH", f"BUG: 末尾是平台摘要就误 FINISH 了 (next={nxt_d})"
print("  ok: 即使 messages[-1] 是平台摘要，提交序号不同仍触发重规划（修掉线上 FINISH）")

print("\n=== 用例 D2：平台把追问合并进摘要（末尾唯一 HumanMessage 即摘要）-> 仍须重规划 ===")
tail_merged = [HumanMessage(content="Here is a summary of the conversation to date:\n## SESSION I\n用户最新追问：对比标普500")]
out_d2 = S.supervisor(make_state(sub_diff=True, tail=tail_merged))
nxt_d2 = out_d2.get("next")
print("  next =", nxt_d2)
assert nxt_d2 and nxt_d2 != "FINISH", f"BUG: 追问被合并进摘要就误 FINISH (next={nxt_d2})"
print("  ok: 追问被合并进摘要（无干净用户消息）-> 回落 FULL 重规划，不 FINISH")


print("\n=== 用例 E：旧 checkpoint（无 _plan_submission_id）续问 -> 仍须重规划 ===")
st_e = make_state(sub_diff=True, tail=[HumanMessage(content="那顺带对比一下标普500呢？")])
st_e["_plan_submission_id"] = None   # 模拟本修复前的旧线程 checkpoint
out_e = S.supervisor(st_e)
nxt_e = out_e.get("next")
print("  next =", nxt_e)
assert nxt_e and nxt_e != "FINISH", f"BUG: 旧线程续问（_plan_submission_id=None）却 FINISH (next={nxt_e})"
print("  ok: 旧线程（无 _plan_submission_id）续问也能触发重规划（不误 FINISH）")


print("\n=== 单测 _classify_followup 意图判定 ===")
assert S._classify_followup("换个说法") == "LIGHT"
assert S._classify_followup("再补充一句行业背景说明") == "LIGHT"
assert S._classify_followup("这个报告能不能更详细一点") == "LIGHT"
assert S._classify_followup("简短点，太长了") == "LIGHT"
assert S._classify_followup("现在再去分析一下标普500") == "FULL"
assert S._classify_followup("帮我写一份新的周报") == "FULL"
assert S._classify_followup("再做一个竞品对比") == "FULL"
assert S._classify_followup("a" * 250) == "FULL"
_orig = S._LIGHT_FOLLOWUP_ON
S._LIGHT_FOLLOWUP_ON = False
assert S._classify_followup("换个说法") == "FULL"
S._LIGHT_FOLLOWUP_ON = _orig
print("  ok: LIGHT/FULL 分类正确，开关可回滚")


print("\n=== 用例 C：LIGHT 轻量微调追问 -> 路由上一轮 agent + 单步计划，不重规划 ===")
msgs_c = [HumanMessage(content="分析纳斯达克100指数近3年资金流向"),
          AIMessage(content="（上一轮的最终报告正文）"),
          HumanMessage(content="换个说法，再补充一句行业背景说明")]
st_c = make_state(sub_diff=True, tail=[HumanMessage(content="换个说法，再补充一句行业背景说明")])
# 让 messages 末条是 LIGHT 追问（覆盖上面默认构造的“对比标普”）
st_c["messages"] = [
    HumanMessage(content="分析纳斯达克100指数近3年资金流向"),
    AIMessage(content="（上一轮的最终报告正文）"),
    HumanMessage(content="换个说法，再补充一句行业背景说明"),
]
st_c["execution_plan"] = [
    PlanStep(title="获取纳指数据", description="x", status="completed"),
    PlanStep(title="chat_agent 撰写最终报告", description="y chat_agent", status="completed"),
]
out_c = S.supervisor(st_c)
nxt_c, plan_c = out_c.get("next"), out_c.get("execution_plan") or []
print("  next =", nxt_c, "| plan 步数 =", len(plan_c), "| current_step =", out_c.get("current_step"))
assert nxt_c == "chat_agent", f"BUG: LIGHT 未路由到上一轮 agent (next={nxt_c})"
assert len(plan_c) == 1 and out_c.get("current_step") == 1
print("  ok: 轻量微调追问走轻量追加（不消耗重规划 LLM）")


print("\n全部断言通过 OK  — supervisor 支持“接着聊”：用提交序号判定续问，"
      "对平台注入摘要免疫（LIGHT 轻量追加 + FULL 整轮重规划），且向后兼容单次任务。")
