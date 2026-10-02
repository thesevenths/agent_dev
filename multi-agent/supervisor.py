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
from planutil import Router, _extract_json_obj, _goal_text, members, _structured_with_retry
from context import _date_context_str
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
            messages = [system_msg] + state["messages"]
            # 跨会话长期记忆注入（首轮规划）：把入口召回的用户背景插在 system 之后、本轮请求之前，
            # 让“计划”本身就贴合用户画像/偏好（如中文报告、金融口径）。为空则不加，零回归。
            _recalled = state.get("recalled_memory") or ""
            if _recalled:
                messages = ([system_msg,
                             HumanMessage(content=("[Long-term memory about the USER, recalled from past "
                                                   "sessions — personalize the plan accordingly; this is "
                                                   "BACKGROUND, not the task]\n" + _recalled))]
                            + list(state["messages"]))

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
                    # 本次 run 启动时刻（仅在首次规划时写入，随 checkpoint 续命）：
                    # 产物幂等守卫用它区分 tmp/ 里"本次 run 的产物"与历史 run 的同号 step 产物。
                    "run_started_at": state.get("run_started_at") or datetime.now().isoformat(),
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
