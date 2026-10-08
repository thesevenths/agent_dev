"""Supervisor 节点：一次性战略规划 + 多轮顺序执行（自适应再规划）。

- 情况1：已有 execution_plan → 每步根据上一步结果审视/改写剩余步骤（_should_replan 决定是否值得
  花一次 LLM；_replan_tail 执行改写，含 #6 早停通道）；
- 情况2：首次遇到请求 → 调 LLM 生成结构化多步计划（只做一次）。
后端（vLLM）不可达时直接终止并给出可操作提示，避免对不堪重负的后端反复横跳形成死循环。
原定义位于 agent.py:1014-1209，拆分时整体迁入。
"""
import os
import logging
from datetime import datetime
from typing import Dict, Any
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from prompt import supervisor_system_prompt
from state import AgentState
from runlog import set_current, ensure_run, log_event, run_file, get_run_id
from plan import (
    _should_replan, _replan_tail, _mark_progress, _plan_view,
    _parse_target_agent, _normalize_plan, _PLAN_GROW_MAX, _render_step_failures,
)
from summary import _summarize_observations, _obs_total, _AGENT_SUMMARY_DISABLE
from planutil import Router, _extract_json_obj, _goal_text, members, _structured_with_retry, _parse_target_agent
from context import _date_context_str
from compress import _compress_messages
from llm import supervisor_llm

logger = logging.getLogger(__name__)

# === E1 终局对账：FINISH 前若存在 failed 步，强制一次 LLM 决策 ===
# 背景（2026-10-02 线上实证）：current >= len(plan) 的分支早于再规划判定，直接 FINISH，
# **完全不调 LLM** —— 失败步既不重试、也不改写、更不会告知用户，只在日志里留一行
# "failed steps: [5]"。用户拿到的是"任务完成了"的假象（step5 邮件实际没发出去）。
# 现在：只要终态计划里还有 failed 步，就花一次 LLM 让模型在三条路里选一条：
#   ① 重试（追加一个 retry 步，需 E2 的长度配额） ② 降级交付 ③ 明确告知用户哪步没做成。
# 设 0 即恢复原「静默 FINISH」语义（零回归）。LLM 不可达/解析失败一律 fail-safe 走原逻辑。
_FINAL_ADJUDICATE = os.environ.get("AGENT_FINAL_ADJUDICATE", "1").lower() in ("1", "true", "yes")


def _extract_longterm(state, memory_key):
    """FINISH 收尾时把本轮蒸馏成跨会话长期记忆（“养龙虾”闭环写入端）。异常吞掉，绝不影响收尾。"""
    try:
        from longterm import extract_and_remember_from_run
        extract_and_remember_from_run(state, memory_key)
    except Exception as e:
        logger.warning(f"[longterm] FINISH extract skipped ({e})")


def _adjudicate_failures(state: dict, plan: list, failed_idx: list, memory_key: str,
                         summary_ctx: str = "") -> dict:
    """FINISH 前的终局对账：存在 failed 步时强制一次 LLM 决策（E1）。

    返回：
      - {"next": <agent>, "execution_plan": ..., "current_step": N, ...} → 追加重试步，继续跑；
      - {"final_note": "..."}                                            → 接受失败，带上说明 FINISH；
      - {}                                                               → 无需干预（走原 FINISH）。
    任何异常一律吞掉并返回 {}，绝不能让对账反过来卡住收尾。
    """
    if not _FINAL_ADJUDICATE or not failed_idx:
        return {}
    try:
        from planutil import Router, members as _members
        goal = state.get("plan_goal") or _goal_text(state)
        grown = int(state.get("plan_grown") or 0)
        quota = max(0, _PLAN_GROW_MAX - grown)
        # 失败原因：优先用结构化档案（step_failures），没有则退化成"仅知道某步 failed"
        fails = state.get("step_failures") or []
        detail = []
        for i in failed_idx:
            rel = [f for f in fails if int(f.get("step") or 0) == i]
            if rel:
                f = rel[-1]
                detail.append(
                    f"  - step {i} (agent={f.get('agent')}, kind={f.get('kind')}): "
                    f"{str(f.get('reason') or '')[:300]}"
                    + (f"\n    how to avoid: {str(f.get('hint') or '')[:200]}"
                       if f.get("hint") else "")
                )
            else:
                st = plan[i - 1] if 0 < i <= len(plan) else {}
                detail.append(f"  - step {i} ({st.get('title', '')}): 标记为 failed（无结构化原因记录）")
        # 无增长配额时不再提供"重试"选项，只让模型在降级/说明之间选，杜绝无限追加。
        if quota > 0:
            rule = (
                f"You may RETRY: return next=<agent> and an 'execution_plan' of length "
                f"{len(plan) + 1} whose LAST step is a retry of the failed step — its description "
                f"MUST encode the failure cause and the concrete constraint that avoids repeating it.\n"
                f"Or return next=\"FINISH\" with 'reason' explaining what was NOT accomplished "
                f"(degraded delivery / tell the user explicitly).\n"
            )
        else:
            rule = (
                "RETRY QUOTA EXHAUSTED — you MUST return next=\"FINISH\". "
                "Use 'reason' to state plainly which step(s) failed and what the user is missing.\n"
            )
        sys_msg = SystemMessage(content=supervisor_system_prompt.replace("{members}", ", ".join(_members)))
        user_msg = HumanMessage(content=(
            f"FINAL ADJUDICATION before finishing this run.\n"
            f"User goal (NEVER change):\n{goal}\n\n"
            f"Plan at finish:\n{_plan_view(plan)}\n\n"
            f"FAILED step(s): {failed_idx}\n" + "\n".join(detail) + "\n\n"
            + (f"Completed steps summary:\n{summary_ctx}\n\n" if summary_ctx else "")
            + f"Artifacts on disk: {state.get('artifacts') or []}\n\n"
            + rule
            + "Return strict JSON with 'next' and 'reason' (plus 'execution_plan' only if retrying)."
        ))
        parsed = _structured_with_retry(supervisor_llm, [sys_msg, user_msg], Router,
                                        label="final-adjudicate")
        if not isinstance(parsed, dict):
            return {}
        nxt = str(parsed.get("next") or "").strip()
        valid = [m.replace("_agent", "") for m in _members] + ["FINISH"]
        if nxt.replace("_agent", "") not in valid:
            return {}
        if nxt == "FINISH":
            note = str(parsed.get("reason") or "").strip()
            log_event(f"[supervisor] final adjudication: accept failure(s) {failed_idx}; "
                      f"note={note[:200]}", memory_key)
            return {"final_note": note}
        rev = _normalize_plan(parsed.get("execution_plan"))
        if not rev:
            return {}
        if len(rev) > len(plan) + quota:
            rev = rev[: len(plan) + quota]
        if len(rev) <= len(plan):
            # 模型没有真正追加重试步 → 无法重做，降级为带说明的 FINISH
            note = str(parsed.get("reason") or "").strip() or \
                f"step {failed_idx} failed and no retry step was produced"
            return {"final_note": note}
        new_plan = rev
        # 关键：把失败步的原因"搬运"到新步号上，否则 handoff 的前向回溯
        # （AGENT_FAIL_LOOKBACK 只看最近 2 步）看不到原失败步的根因，重试必然再踩同一个坑。
        carried = list(state.get("step_failures") or [])
        new_step_no = len(new_plan)
        for i in failed_idx:
            rel = [f for f in fails if int(f.get("step") or 0) == i]
            for f in rel:
                g = dict(f)
                g["step"] = new_step_no
                g["kind"] = f"retry-of-{i}"
                carried.append(g)
        log_event(
            f"[supervisor] final adjudication: RETRY step {failed_idx} → appended as step "
            f"{new_step_no} ({new_plan[-1].get('title', '')}); "
            f"plan_grown {grown} -> {grown + (len(new_plan) - len(plan))}",
            memory_key,
        )
        return {
            "next": nxt,
            "reason": f"Final adjudication: retrying failed step {failed_idx} as step {new_step_no}.",
            "current_step": len(plan),          # 指向新追加的重试步
            "execution_plan": new_plan,
            "plan_goal": state.get("plan_goal"),
            "run_started_at": state.get("run_started_at"),
            "plan_grown": grown + (len(new_plan) - len(plan)),
            "step_failures": carried[-20:],
            "replan_noop_streak": 0,
        }
    except Exception as e:
        logger.warning(f"[supervisor] final adjudication skipped ({e}); finishing as-is")
        return {}


# === 同线程“接着聊”支持（2026-10-07 实证修复）===
# 背景：旧逻辑在 current_step>=len(plan) 时无条件 FINISH，导致同线程续问被完全忽略——
# 用户首问“分析纳斯达克100”跑完后，再发一条追问，supervisor 只看已完成计划直接收尾，
# 新消息从未被处理（日志铁证：续问 run 直接打印 “FINISH: all steps ... completed”）。
# 修复：计划已完成且最后一条消息是一条“新的用户 HumanMessage”（非本系统注入的背景/任务消息）
# 时，清空旧计划、落到“情况2”按新请求重新规划，保留 messages 历史作为上下文。
# 判定用的前缀清单：agents.py 给首条用户消息前缀的日期上下文、跨会话记忆注入、上游步骤摘要、
# 子 agent 任务下发——这些都不是用户真实输入，绝不误判为追问。
_INJECT_PREFIXES = (
    "[System context]",       # agents.py 给首条用户消息前缀的日期/时段上下文
    "[Long-term memory",      # 跨会话长期记忆注入（run_start 召回结果）
    "[Summary of completed",  # 上游已完成步摘要
    "Assignment:",            # 子 agent 任务下发（_step_assignment_text）
    "Here is a summary",      # langgraph dev 线程恢复时注入的“对话摘要”合成消息
    "Here's a summary",       # 同上（口语化变体）
    "Here is a summary of the conversation",  # 同上（完整前缀）
)


def _starts_with_inject_prefix(c: str) -> bool:
    """内容是否以系统/平台注入背景前缀开头（用于识别非用户真实输入）。"""
    c = (c or "").strip()
    return any(c.startswith(p) for p in _INJECT_PREFIXES)


def _is_new_user_followup(msg) -> bool:
    """最后一条消息是否是一条“用户新追问”（而非系统注入背景/任务消息）。"""
    if not isinstance(msg, HumanMessage):
        return False
    c = msg.content if isinstance(msg.content, str) else str(msg.content)
    c = c.strip()
    if not c:
        return False
    return not _starts_with_inject_prefix(c)


def _extract_followup_text(messages) -> str:
    """从线程消息里抽取“用户真实的新追问文本”，用于 LIGHT/FULL 意图分类与轻量微调 step 描述。

    采用「最后一条非工具 AIMessage 之后的第一条非注入用户消息」——即使 langgraph dev 在末尾注入了
    “对话摘要”合成消息也能跳过它、拿到用户真正的追问（若平台把追问原文作为独立 HumanMessage 附在
    摘要之前）；若平台把追问合并进摘要（唯一 HumanMessage 就是摘要）则返回空串（调用方回落 FULL 重规划）。
    """
    msgs = messages or []
    last_ai = -1
    for i, m in enumerate(msgs):
        if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
            last_ai = i
    window = msgs[last_ai + 1:] if last_ai >= 0 else msgs
    for m in window:
        if isinstance(m, HumanMessage):
            c = m.content if isinstance(m.content, str) else str(m.content)
            if not _starts_with_inject_prefix(c):
                return c.strip()
    # 兜底：后续窗口里找不到非注入用户消息（如平台把追问合并进末尾摘要）→ 返回空串，
    # 调用方回落为 FULL 整轮重规划（安全，绝不误走 LIGHT 把原任务当微调）。
    return ""


# === 同线程“轻量追加”意图判定（2026-10-07 新增）===
# 背景：情况0 对“任何新追问”一律整轮全量重规划（一次 LLM 规划调用）。但很多续问只是对上一轮
# 产物的微调（“换个说法”“再补充一句”“简短点”“详细点”），完全不需要重做整个任务。这类轻量追问
# 直接路由给“上一轮最后一个执行的 agent”，挂一个单步计划（不会触发再规划 LLM，见 plan._should_replan
# 对 current<=0 直接跳过），由该 agent 基于 messages 历史里的已有产出直接改写/补一句即可，
# 省掉整轮重规划开销。判定纯关键词启发式、零 LLM、确定性、可经开关回滚到改造前行为。
_LIGHT_FOLLOWUP_ON = os.environ.get("AGENT_LIGHT_FOLLOWUP", "1").lower() in ("1", "true", "yes")
_LIGHT_FOLLOWUP_MAXLEN = int(os.environ.get("AGENT_LIGHT_FOLLOWUP_MAXLEN", "200") or 200)
# 正向信号：命中即“候选 LIGHT”。覆盖中英文常见微调措辞（加一句/改写/润色/长短/口吻等）。
_LIGHT_POSITIVE = (
    "换个说法", "换种说法", "换一种说法", "重新表述", "换种表达", "重新表达", "用更",
    "再补充", "补充一句", "加一句", "再写一句", "润色", "改写", "措辞", "语气",
    "口吻", "简短点", "更短一点", "简短文", "太长了", "精简", "详细点", "更详细",
    "展开说", "换个角度", "换个方式", "换种风格", "重新组织", "重新写", "重新生成一下",
    "重新措辞", "措辞能不能", "这样写", "换个写法", "换一种表达", "说得通俗", "更口语",
    "rephrase", "reword", "put it differently", "in other words", "add a sentence",
    "append", "shorten", "make it shorter", "more concise", "more detail", "rewrite",
    "word it", "polish", "tone",
)
# 负向信号：命中即“强制 FULL 重规划”，绝不走轻量，避免把新任务误判成微调。
_LIGHT_NEGATIVE = (
    "现在去", "帮我做", "再做一个", "再分析", "重新分析", "分析一下", "写一份",
    "写一个新", "生成一份", "生成一个新的", "创建一个", "查一下", "查一查", "搜索",
    "搜一下", "爬取", "抓取", "对比", "比较", "总结一下", "翻译", "帮我写", "新任务",
    "换一个主题", "另一个", "新的报告", "再做", "重新来", "换个项目", "重新做",
    "now go", "analyze", "create a", "write a", "generate a new", "build a",
)


def _classify_followup(text: str) -> str:
    """把一条新追问分类为 'LIGHT'（轻量微调，挂单步计划路由到上一轮 agent）或
    'FULL'（新任务/大改，整轮全量重规划）。

    判定顺序（确定性、零 LLM）：
      1) 总开关关闭 -> FULL（= 改造前行为，零回归）；
      2) 命中负向信号 -> FULL（新任务优先，绝不误走轻量）；
      3) 长度超门（默认 200 字）-> FULL（长文几乎都是新任务）；
      4) 命中正向信号 -> LIGHT；
      5) 其余 -> FULL（保守默认，宁可多花一次规划也不误判）。
    """
    if not _LIGHT_FOLLOWUP_ON:
        return "FULL"
    t = (text or "").strip().lower()
    if not t:
        return "FULL"
    if any(k in t for k in _LIGHT_NEGATIVE):
        return "FULL"
    if len(t) > _LIGHT_FOLLOWUP_MAXLEN:
        return "FULL"
    if any(k in t for k in _LIGHT_POSITIVE):
        return "LIGHT"
    return "FULL"


def _classify_supervisor_error(e: Exception) -> str:
    """把 supervisor 调用后端的异常分类，避免把“上下文超长/参数被拒”一律误报成“后端连不上”。

    返回 'context_overflow' | 'unreachable' | 'other'。纯字符串匹配、零依赖、绝不抛。
    """
    try:
        s = f"{type(e).__name__}: {e}".lower()
    except Exception:
        return "other"
    if any(k in s for k in (
        "maximum context length", "context_length_exceeded", "input_tokens",
        "reduce the length of the input", "context length", "too many tokens",
    )):
        return "context_overflow"
    if any(k in s for k in (
        "connection error", "connectionerror", "connection refused", "connection aborted",
        "failed to establish", "max retries", "timeout", "timed out", "read timed out",
        "getaddrinfo", "name or service not known", "could not connect", "unreachable",
        "ssl", "502 bad gateway", "503 service", "504 gateway", "peer closed connection",
    )):
        return "unreachable"
    return "other"


def supervisor(state: AgentState) -> Dict[str, Any]:
    """Supervisor：支持一次性规划 + 多轮顺序执行；同线程可“接着聊”（见上方检测）"""
    try:
        # 让本轮运行的关键日志（含子 agent 内触发的 tavily 检索）落到正确的运行日志文件。
        set_current(state.get("memory_key"))
        # 情况0：同线程“接着聊”——计划已完成，且这是一次【新的用户提交】。
        # 判定用「提交序号」而非「messages[-1] 是否为新用户消息」：langgraph dev 续跑线程时会在
        # messages 末尾注入“对话摘要”合成消息（"Here is a summary of the conversation to date..."），
        # 导致 messages[-1] 不可靠。改为比较 run_start 每次提交生成的 _submission_id 与计划创建时
        # 记录的 _plan_submission_id —— 二者不同即说明是续问，与平台注入无关，稳定。
        plan = state.get("execution_plan") or []
        if plan and len(plan) > 0 and state.get("current_step", 0) >= len(plan):
            _sub = state.get("_submission_id")
            _plan_sub = state.get("_plan_submission_id")
            # 旧 checkpoint（本修复前）没有 _plan_submission_id 字段（为 None）；
            # 此时一律视为“新提交”（None 视作与任何新序号都不同），避免旧线程续问仍被误 FINISH。
            if _sub and _sub != (_plan_sub or ""):
                mk = state.get("memory_key")
                _txt = _extract_followup_text(state.get("messages") or [])
                intent = _classify_followup(_txt) if _txt else "FULL"
                if intent == "LIGHT":
                    # 轻量追加：直接路由到上一轮最后一个执行的 agent，挂一个单步计划。
                    # 该单步计划 current_step=1、len=1，下一轮 supervisor 进入 FINISH 分支，
                    # 且 plan._should_replan 对 current<=0 直接跳过 —— 全程不触发任何额外规划 LLM。
                    last_agent = _parse_target_agent(plan[-1]) if plan else "context_engineer_agent"
                    if last_agent not in members:
                        last_agent = "context_engineer_agent"
                    step = {
                        "title": "按用户追加请求微调上一轮产物",
                        "description": (
                            "用户追加请求（轻量微调，不要重新规划整个任务）：\n" + (_txt or "(见对话历史中的最新追问)") +
                            "\n\n这是对该会话上一轮已完成的答复/产物的微调，不要重做整个任务；"
                            "直接基于 messages 历史里已有的产出（上一轮的答复/落盘文件）进行改写、"
                            "补充或润色后回复用户。"
                        ),
                        "status": "pending",
                    }
                    log_event(
                        f"[supervisor] 检测到同线程轻量微调追问（LIGHT, 提交 {_sub}≠{_plan_sub}）"
                        f"-> 路由到 {last_agent}，挂单步计划，跳过全量重规划",
                        mk,
                    )
                    return {
                        "next": last_agent,
                        "reason": f"Light follow-up (refine previous output): route to {last_agent} without full re-plan.",
                        "execution_plan": [step],
                        "plan_goal": state.get("plan_goal"),
                        # 新提交 = 新任务边界：必须重置 run 启动时刻，否则幂等守卫会把
                        # 上一轮遗留的同号 step 产物误认成“本 run 已产出”而跳过 LIGHT 步。
                        "run_started_at": datetime.now().isoformat(),
                        "current_step": 1,
                        "_plan_submission_id": _sub,
                    }
                # FULL：清空旧计划，落到下方情况2 重新规划（= 原逻辑）
                log_event(
                    f"[supervisor] 检测到同线程新追问（提交 {_sub}≠{_plan_sub}）"
                    f"-> 清空旧计划，按新请求重新规划",
                    mk,
                )
                plan = []   # 落到情况2
        # 情况1：已有执行计划 → 自适应再规划（每步根据上一步结果审视/改写剩余步骤）
        if plan and len(plan) > 0:
            memory_key = state.get("memory_key")
            ensure_run(memory_key)
            current = state.get("current_step", 0)
            if current >= len(plan):
                # 所有步骤都完成了。
                # 关键：必须把最后一步（以及任何遗留步）标 completed 并**写回 state**。
                # 原来的实现在这里直接 return 且不带 execution_plan，而 completed 标记是靠
                # _mark_progress 在"下一轮 supervisor"写的 —— 最后一轮走 FINISH 就没人再写，
                # 导致终态 execution_plan 里最后一步永远是 pending，看不出这次运行到底做完没有。
                final_plan = _mark_progress(plan, len(plan))
                failed_idx = [i + 1 for i, s in enumerate(final_plan) if s.get("status") == "failed"]
                # E1 终局对账：有 failed 步时不再静默 FINISH，先让模型在
                # 重试 / 降级交付 / 明确告知 之间做一次决策。
                adj = _adjudicate_failures(state, final_plan, failed_idx, memory_key) \
                    if failed_idx else {}
                if adj.get("next") and str(adj.get("next")) != "FINISH":
                    # 追加重试步 → 本轮尚未结束，不抽取长期记忆（避免把半成品蒸馏入库）
                    return adj
                note = str(adj.get("final_note") or "")
                log_event(
                    "[supervisor] FINISH: all steps in execution plan completed."
                    + (f" (failed steps: {failed_idx})" if failed_idx else "")
                    + (f"\n[FINAL NOTE] {note}" if note else "")
                    + f"\n{_plan_view(final_plan)}",
                    memory_key,
                )
                # 长期记忆抽取（写入端）：本轮真正结束，用 1 次 LLM 蒸馏跨会话记忆入库。
                _extract_longterm(state, memory_key)
                return {
                    "next": "FINISH",
                    "reason": (f"Finished with {len(failed_idx)} failed step(s): {note}"
                               if note else "All tasks in execution plan completed."),
                    "current_step": current,
                    "execution_plan": final_plan,
                    "plan_goal": state.get("plan_goal"),
                    "run_started_at": state.get("run_started_at"),
                    "final_note": note,
                }

            step_text = plan[current]
            target_agent = _parse_target_agent(step_text)
            # 保存"原计划给本步指定的 agent"，供下方空转判定比较：再规划可能只换了执行者
            # （plan 文本一字未改），那也是一次有效决策，不能记成空转。
            planned_agent = target_agent
            plan_before = _mark_progress(plan, current)
            # 跨步语义摘要：把已完成步的 observations 提炼成要点清单（落盘 log/<thread>_summary.md），
            # 作为再规划与下游 agent 的语义上下文；LLM 不可达时 _summarize_observations 内部回落提取式摘要。
            # 注意：这是**增量滚动摘要**，无新增产出时直接复用旧清单，不再每轮全量重述历史。
            summary_ctx = ""
            if not _AGENT_SUMMARY_DISABLE:
                summary_ctx = _summarize_observations(state)
                if summary_ctx:
                    log_event(f"[supervisor] completed-steps summary (fed to re-plan + downstream):\n{summary_ctx}", memory_key)
            # 早停决策必须由模型显式截断计划来声明，故再规划调用在"最后一步"仍然保留（价值高）；
            # 其余情形下若判断为空转则整轮跳过，省下一次 LLM 调用。
            should, why = _should_replan(state, plan, current)
            if not should:
                plan = _mark_progress(plan, current)
                log_event(
                    f"[supervisor] re-plan SKIPPED for step {current + 1}/{len(plan)}: {why}\n"
                    f"{_plan_view(plan, current)}\n=> next={target_agent} (from original plan)",
                    memory_key,
                )
                return {
                    "next": target_agent,
                    "reason": f"Following execution plan step {current + 1}/{len(plan)}: {step_text} (re-plan skipped: {why})",
                    "execution_plan": plan,
                    "plan_goal": state.get("plan_goal"),
                    "run_started_at": state.get("run_started_at"),
                    "plan_summary": summary_ctx,
                    "summary_obs_seen": _obs_total(state),
                    # 关键：跳过时**必须原样带回**空转计数。若这里不写（或写成 0），
                    # 计数会在跳过那轮被清零，下一轮又重新启用再规划 → 退化成
                    # "规划/规划/跳过" 的锯齿，只省掉 1/3 的调用而不是全部。
                    "replan_noop_streak": int(state.get("replan_noop_streak") or 0),
                    "current_step": current + 1,
                }
            try:
                # 调用 LLM 审视剩余步骤（可跳过/改写已失效步骤），带防死循环硬约束
                target_agent, plan, finish_reason = _replan_tail(state, plan, current, summary_ctx=summary_ctx)
                # 空转检测：BEFORE==AFTER 说明这次 LLM 调用没有任何收益，累计到阈值后自动停用。
                # 注意：只比 plan 文本会把"plan 未改但换了执行者"误记成空转 —— 2026-10-02 step5 就是
                # plan 一字未改、却把 chat_agent(发邮件) 换成了 code_agent(重做报告)，这是一次真实决策
                # （虽然该决策本身有问题），若记成空转会累积 streak 进而停用后续再规划。
                def _norm_agent(a):
                    # 同时兼容 "chat_agent"（plan 解析结果）与 "ChatAgent"（agent.name）两种写法
                    t = str(a or "").strip().lower().replace("_agent", "")
                    if t.endswith("agent") and len(t) > len("agent"):
                        t = t[: -len("agent")]
                    return t

                agent_changed = current < len(plan_before) and (
                    _norm_agent(target_agent) != _norm_agent(planned_agent))
                changed = (_plan_view(plan_before, current) != _plan_view(plan, current)) or agent_changed
                streak = 0 if changed else (int(state.get("replan_noop_streak") or 0) + 1)
                if not changed:
                    logger.warning(
                        f"supervisor re-plan was a NO-OP (BEFORE==AFTER) at step {current + 1}; "
                        f"noop streak={streak}"
                    )
                elif agent_changed and _plan_view(plan_before, current) == _plan_view(plan, current):
                    logger.info(
                        f"supervisor re-plan changed ONLY the executor at step {current + 1}: "
                        f"{planned_agent} -> {target_agent} (plan text unchanged)"
                    )
                if finish_reason is not None:
                    # 早停：模型显式截断了计划 → 剩余步骤作废，直接收尾（#6）
                    final_plan = _mark_progress(plan, len(plan))
                    log_event(
                        f"[supervisor] EARLY FINISH at step {current + 1}/{len(plan_before)} — "
                        f"{len(plan_before) - len(final_plan)} remaining step(s) dropped: {finish_reason}\n"
                        f"{_plan_view(final_plan)}",
                        memory_key,
                    )
                    # 长期记忆抽取（写入端）：早停同样是一次 run 结束，蒸馏跨会话记忆入库。
                    _extract_longterm(state, memory_key)
                    return {
                        "next": "FINISH",
                        "reason": f"Early finish after {len(final_plan)}/{len(plan_before)} steps: {finish_reason}",
                        "current_step": len(final_plan),
                        "execution_plan": final_plan,
                        "plan_goal": state.get("plan_goal"),
                        "run_started_at": state.get("run_started_at"),
                        "plan_summary": summary_ctx,
                        "summary_obs_seen": _obs_total(state),
                        "replan_noop_streak": streak,
                    }
                log_event(
                    f"[supervisor] re-planning step {current + 1}/{len(plan_before)}:\n"
                    f"BEFORE:\n{_plan_view(plan_before, current)}\n"
                    f"AFTER:\n{_plan_view(plan)}\n=> next={target_agent}"
                    + ("" if changed else "\n[NOTE] re-plan was a NO-OP: AFTER identical to BEFORE (LLM cost wasted)"),
                    memory_key,
                )
            except Exception as re:
                logger.warning(f"supervisor re-plan failed ({re}); follow original plan step {current + 1}")
                # 回落：保持原计划仅推进当前步，但 status 必须照样反映进度（不依赖 LLM）
                target_agent, plan = target_agent, _mark_progress(plan, current)
                streak = int(state.get("replan_noop_streak") or 0)
                log_event(
                    f"[supervisor] re-plan FAILED ({re}); following original step {current + 1}/{len(plan)}:\n"
                    f"{_plan_view(plan, current)}",
                    memory_key,
                )
            return {
                "next": target_agent,
                "reason": f"Following execution plan step {current + 1}/{len(plan)}: {step_text}",
                "execution_plan": plan,
                "plan_goal": state.get("plan_goal"),  # 原样带回，再规划不改 goal
                "run_started_at": state.get("run_started_at"),  # 原样带回，供产物幂等守卫跨进程续跑时仍能区分历史产物
                "plan_summary": summary_ctx,          # 跨步语义摘要，随 state 下发给子 agent
                "summary_obs_seen": _obs_total(state),  # 摘要游标：标记这些 observations 已折叠进 plan_summary
                "replan_noop_streak": streak,         # 空转计数：连续多次无效后停用再规划
                # E2：计划长度增长计数（只在真正变长时累加，用于约束「重做载体」配额）
                "plan_grown": int(state.get("plan_grown") or 0)
                              + max(0, len(plan) - len(plan_before)),
                "current_step": current + 1   # 关键：推进进度
            }

        # 情况2：第一次遇到用户请求 → 做战略规划（只做一次）
        else:
            # 注意：supervisor_system_prompt 内含 JSON 示例（大量 { }），str.format 会把它们当成占位符而抛 KeyError。
            # 该模板唯一占位符是 {members}，用 replace 安全替换，避免转义整段 JSON 大括号。
            system_msg = SystemMessage(content=supervisor_system_prompt.replace(
                "{members}", ", ".join(members)
            ) + "\n\n" + _date_context_str())
            # 关键：给 supervisor 首轮规划也套上与子 agent 同一道防溢出保险 _compress_messages。
            # 之前这里把整条线程 state["messages"]（跨很多轮累加、被 checkpoint 持久化）原样塞进一个
            # prompt，长线程会把输入顶到模型上限（实证 2026-10-08：262145 > 262144 → vLLM 回 400）。
            # 规划只需「目标 + 最近几轮 + 摘要」，不需全量原始历史；_compress_messages 保首条用户 query
            # + 最近若干条，并对超长 Tool/AI 消息做语义提炼（时间/地点/人物/事件/ID/IP/路径/数值等核心
            # 要素一律保留，见 compress._CONDENSE_PROMPT），既防溢出又不丢要点。
            _hist = _compress_messages(state["messages"])
            messages = [system_msg] + _hist
            # 跨会话长期记忆注入（首轮规划）：把入口召回的用户背景插在 system 之后、本轮请求之前，
            # 让“计划”本身就贴合用户画像/偏好（如中文报告、金融口径）。为空则不加，零回归。
            _recalled = state.get("recalled_memory") or ""
            if _recalled:
                messages = ([system_msg,
                             HumanMessage(content=("[Long-term memory about the USER, recalled from past "
                                                   "sessions — personalize the plan accordingly; this is "
                                                   "BACKGROUND, not the task]\n" + _recalled))]
                            + _hist)

            # 优先结构化输出；本地 vLLM 对 TypedDict 结构化输出支持不稳定 → 先带错误重试，
            # 全败才回落裸调用+手动抽 JSON（鲁棒性#3）。
            parsed = _structured_with_retry(supervisor_llm, messages, Router, label="first-plan")
            if not isinstance(parsed, dict):
                ai = supervisor_llm.invoke(messages)
                content = ai.content if isinstance(ai, AIMessage) else str(ai)
                parsed = _extract_json_obj(content)
            response = parsed or {}

            # 如果模型给出了计划，就采纳（归一化为结构化步骤，每步 status=pending）
            plan = _normalize_plan(response.get("execution_plan"))
            if plan:
                goal = response.get("goal") or _goal_text(state)
                memory_key = state.get("memory_key")
                # 日志文件已由图入口节点 run_start 统一开好（唯一 run_id），此处确保存在即可，不重复开文件。
                ensure_run(memory_key)
                logger.info(f"Supervisor created execution plan (goal={goal!r}):\n" + "\n".join(
                    f"{i+1}. {s.get('title','')}: {s.get('description','')}" for i, s in enumerate(plan)
                ))
                log_event(
                    f"[supervisor] created execution plan (goal={goal!r}):\n{_plan_view(plan)}\n"
                    f"[supervisor] run log file: {run_file(get_run_id())}",
                    memory_key,
                )
                # 第一步立刻执行
                first_agent = _parse_target_agent(plan[0])
                return {
                    "next": first_agent,
                    "reason": f"Starting execution plan step 1/{len(plan)}: {plan[0].get('description','')}",
                    "execution_plan": plan,
                    "plan_goal": goal,
                    "current_step": 1,
                    # 记录本次规划对应的提交序号，供「情况0 接着聊」判定续问（_submission_id 变化时即为新提交）
                    "_plan_submission_id": state.get("_submission_id"),
                    # 新计划 = 新任务边界，以下字段必须重置（2026-10-07 23:21 线上实证三连 bug）：
                    # 1) run_started_at 若沿用旧值（旧实现 `state.get(...) or now`），幂等守卫
                    #    （agents.py 按 *__stepN__* + mtime≥start 匹配）会把【上一个问题】计划留下的
                    #    同号产物误认成【本问题】已产出 → 新计划各步被整排假跳过；
                    # 2) 旧 plan_summary/observations 带着【旧计划步号】喂进再规划/早停 LLM，
                    #    导致“Step 6 已发邮件”式幻觉触发 EARLY FINISH（新计划根本只有 5 步）。
                    #    历史产物文件仍在盘上、完整对话仍在 messages 里，清空不丢信息。
                    # 注意：同一次提交的中途崩溃续跑不走本分支（plan 非空→情况1），
                    # 幂等守卫对真·断点续跑的保护不受影响。
                    "run_started_at": datetime.now().isoformat(),
                    "plan_summary": None,
                    "observations": [],
                    "obs_total": 0,
                    "summary_obs_seen": 0,
                    "artifacts": [],
                    "replan_noop_streak": 0,
                    "step_failures": [],
                }
            else:
                # 降级为传统单轮路由（兼容旧逻辑）
                return {
                    "next": response.get("next", "context_engineer_agent"),
                    "reason": (response.get("reason", "") or "") + " (no multi-step plan generated)"
                }

    except Exception as e:
        logger.error(f"Supervisor error: {e}")
        ec = state.get("error_count", 0) + 1
        # 分类归因：避免把“上下文超长/参数被拒”一律误报成“后端连不上”（实证 2026-10-08 那次 400
        # 是输入 token 超模型上限，后端其实在线）。三类各给准确、可操作的提示。若真·不可达，
        # 绝不能盲目路由到同样依赖该后端的子 agent，否则反复横跳、对过载后端形成正反馈 → 直接收尾。
        kind = _classify_supervisor_error(e)
        if kind == "context_overflow":
            reason = f"Supervisor failed: context length exceeded ({e}); backend reachable, input too long."
            detail = (
                "⚠️ 上下文超长：后端模型在线，但本次请求的输入 token 超过了模型上限，被以 400 拒绝"
                "（这不是连不上）。\n"
                f"底层错误：{e}\n\n"
                "处理方向：\n"
                "1. 这条线程历史太长（messages 跨很多轮累加）——最省事：点顶部 + 新开一个 thread 再继续。\n"
                "2. 单条工具/文件内容过大——让相关 agent 把大产物落盘到 tmp/，只回传路径而非全文。\n"
                "3. supervisor 首轮规划已接入 _compress_messages 防溢出；若仍触发，多为单条消息本身超限，"
                "可调低 AGENT_CTX_TOOL_CHARS / AGENT_CTX_AI_CHARS 或缩小单次读取范围。"
            )
        elif kind == "unreachable":
            reason = f"Supervisor failed ({e}); backend LLM unreachable, stopping to avoid infinite loop."
            detail = (
                "⚠️ 调度器（supervisor）无法连接后端模型服务，已停止本次任务以避免死循环。\n"
                f"底层错误：{e}\n\n"
                "排查方向（修复后端后重新提交即可）：\n"
                "1. 在 .env 的 LLM_BASE_URL 配置的 vLLM 服务是否在线/已崩溃（这是最常见原因）。\n"
                "2. 网络是否通畅；当前 LLM_VERIFY_SSL=false 已跳过证书校验，连不上多为地址或网络问题。\n"
                "3. vLLM 是否过载被拒连——并发过高时也会批量 Connection error，稍后重试或扩容。"
            )
        else:
            reason = f"Supervisor failed ({e}); stopping to avoid infinite loop."
            detail = (
                "⚠️ 调度器（supervisor）调用后端模型失败，已停止本次任务以避免死循环。\n"
                f"底层错误：{e}\n\n"
                "排查方向：确认后端在线、模型名/参数正确、请求未超出上下文上限；修复后重新提交即可。"
            )
        return {
            "next": "FINISH",
            "reason": reason,
            "error_count": ec,
            "messages": [AIMessage(content=detail)],
        }
