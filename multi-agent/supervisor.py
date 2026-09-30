"""Supervisor 节点：一次性战略规划 + 多轮顺序执行（自适应再规划）。

- 情况1：已有 execution_plan → 每步根据上一步结果审视/改写剩余步骤（_should_replan 决定是否值得
  花一次 LLM；_replan_tail 执行改写，含 #6 早停通道）；
- 情况2：首次遇到请求 → 调 LLM 生成结构化多步计划（只做一次）。
后端（vLLM）不可达时直接终止并给出可操作提示，避免对不堪重负的后端反复横跳形成死循环。
原定义位于 agent.py:1014-1209，拆分时整体迁入。
"""
import logging
from typing import Dict, Any
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from prompt import supervisor_system_prompt
from state import AgentState
from runlog import set_current, ensure_run, start_run, log_event
from plan import (
    _should_replan, _replan_tail, _mark_progress, _plan_view,
    _parse_target_agent, _normalize_plan,
)
from summary import _summarize_observations, _obs_total, _AGENT_SUMMARY_DISABLE
from planutil import Router, _extract_json_obj, _goal_text, members
from context import _date_context_str
from llm import supervisor_llm

logger = logging.getLogger(__name__)


def supervisor(state: AgentState) -> Dict[str, Any]:
    """Supervisor：支持一次性规划 + 多轮顺序执行"""
    try:
        # 让本轮运行的关键日志（含子 agent 内触发的 tavily 检索）落到正确的运行日志文件。
        set_current(state.get("memory_key"))
        # 情况1：已有执行计划 → 自适应再规划（每步根据上一步结果审视/改写剩余步骤）
        plan = state.get("execution_plan") or []
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
                log_event(
                    "[supervisor] FINISH: all steps in execution plan completed."
                    + (f" (failed steps: {failed_idx})" if failed_idx else "")
                    + f"\n{_plan_view(final_plan)}",
                    memory_key,
                )
                return {
                    "next": "FINISH",
                    "reason": "All tasks in execution plan completed.",
                    "current_step": current,
                    "execution_plan": final_plan,
                    "plan_goal": state.get("plan_goal"),
                }

            step_text = plan[current]
            target_agent = _parse_target_agent(step_text)
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
                # 空转检测：BEFORE==AFTER 说明这次 LLM 调用没有任何收益，累计到阈值后自动停用
                changed = _plan_view(plan_before, current) != _plan_view(plan, current)
                streak = 0 if changed else (int(state.get("replan_noop_streak") or 0) + 1)
                if not changed:
                    logger.warning(
                        f"supervisor re-plan was a NO-OP (BEFORE==AFTER) at step {current + 1}; "
                        f"noop streak={streak}"
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
                    return {
                        "next": "FINISH",
                        "reason": f"Early finish after {len(final_plan)}/{len(plan_before)} steps: {finish_reason}",
                        "current_step": len(final_plan),
                        "execution_plan": final_plan,
                        "plan_goal": state.get("plan_goal"),
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
                "plan_summary": summary_ctx,          # 跨步语义摘要，随 state 下发给子 agent
                "summary_obs_seen": _obs_total(state),  # 摘要游标：标记这些 observations 已折叠进 plan_summary
                "replan_noop_streak": streak,         # 空转计数：连续多次无效后停用再规划
                "current_step": current + 1   # 关键：推进进度
            }

        # 情况2：第一次遇到用户请求 → 做战略规划（只做一次）
        else:
            # 注意：supervisor_system_prompt 内含 JSON 示例（大量 { }），str.format 会把它们当成占位符而抛 KeyError。
            # 该模板唯一占位符是 {members}，用 replace 安全替换，避免转义整段 JSON 大括号。
            system_msg = SystemMessage(content=supervisor_system_prompt.replace(
                "{members}", ", ".join(members)
            ) + "\n\n" + _date_context_str())
            messages = [system_msg] + state["messages"]

            # 优先结构化输出；本地 vLLM 对 TypedDict 结构化输出支持不稳定，失败则手动解析 content
            parsed = None
            try:
                resp = supervisor_llm.with_structured_output(Router).invoke(messages)
                parsed = dict(resp) if resp is not None else None
            except Exception as se:
                logger.warning(f"supervisor structured output failed ({se}); fallback to manual JSON parse")
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
                run_path = start_run(memory_key)  # 新运行：开一个带时间戳的日志文件
                logger.info(f"Supervisor created execution plan (goal={goal!r}):\n" + "\n".join(
                    f"{i+1}. {s.get('title','')}: {s.get('description','')}" for i, s in enumerate(plan)
                ))
                log_event(
                    f"[supervisor] created execution plan (goal={goal!r}):\n{_plan_view(plan)}\n"
                    f"[supervisor] run log file: {run_path}",
                    memory_key,
                )
                # 第一步立刻执行
                first_agent = _parse_target_agent(plan[0])
                return {
                    "next": first_agent,
                    "reason": f"Starting execution plan step 1/{len(plan)}: {plan[0].get('description','')}",
                    "execution_plan": plan,
                    "plan_goal": goal,
                    "current_step": 1
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
        # 后端（vLLM）不可达时，supervisor 自身无法完成规划；若按旧逻辑盲目路由到
        # context_engineer_agent（它同样依赖该后端去调用 LLM），二者会反复横跳、永不终止，
        # 并持续对已经不堪重负的后端发起连接、形成正反馈。故直接终止并给出可操作提示。
        return {
            "next": "FINISH",
            "reason": f"Supervisor failed ({e}); backend LLM unreachable, stopping to avoid infinite loop.",
            "error_count": ec,
            "messages": [AIMessage(content=(
                "⚠️ 调度器（supervisor）无法连接后端模型服务，已停止本次任务以避免死循环。\n"
                f"底层错误：{e}\n\n"
                "排查方向（修复后端后重新提交即可）：\n"
                "1. 在 .env 的 LLM_BASE_URL 配置的 vLLM 服务是否在线/已崩溃（这是最常见原因）。\n"
                "2. 网络是否通畅；当前 LLM_VERIFY_SSL=false 已跳过证书校验，连不上多为地址或网络问题。\n"
                "3. vLLM 是否过载被拒连——并发过高时也会批量 Connection error，稍后重试或扩容。"
            ))],
        }
