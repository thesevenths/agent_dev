"""计划改写 + 进度标记 + 自适应重规划引擎（re-plan）。

含：
  - 进度/状态维护：_mark_progress / _mark_failed / _plan_view；
  - 再规划决策：_should_replan（证据门 + 空转停用）+ _replan_evidence（#4 有证据才再规划）；
  - 再规划执行：_replan_tail（调用 LLM 审视并改写剩余步骤，含 #6 早停通道）。
原定义位于 agent.py:781-1013，开关位于 1237-1250，拆分时整体迁入。
依赖：planutil（解析/introspection）、summary（_summarize_observations）、context（日期上下文）、
compress（_msg_text）、llm（supervisor_llm）、prompt（supervisor_system_prompt）。
"""
import os
import re
import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from llm import supervisor_llm
from prompt import supervisor_system_prompt
from planutil import (
    Router, _extract_json_obj, _normalize_plan, _parse_target_agent,
    _goal_text, _build_replan_context, members, _structured_with_retry,
)
from summary import _summarize_observations, _AGENT_SUMMARY_DISABLE
from context import _date_context_str
from compress import _msg_text
from handoff import _render_step_failures

logger = logging.getLogger(__name__)

# 再规划（re-plan）总开关 + 空转停用阈值：
# 线上实证多轮 BEFORE==AFTER 纯回显（LLM 开销白花），连续空转达阈值后自动停用后续再规划。
_AGENT_REPLAN_DISABLE = os.environ.get("AGENT_REPLAN_DISABLE", "").lower() in ("1", "true", "yes")
_REPLAN_NOOP_MAX = int(os.environ.get("AGENT_REPLAN_NOOP_MAX", "2"))
# 再规划"证据"信号集：最近观察里出现这些强失败/中断信号才认为计划需改写（对应 #4 有证据才再规划）。
# 可用 AGENT_REPLAN_SIGNALS 以逗号覆盖（空值则用下方默认集）。
_AGENT_REPLAN_SIGNALS_ENV = os.environ.get("AGENT_REPLAN_SIGNALS", "")
_REPLAN_EVIDENCE_SIGNALS = [s.strip() for s in _AGENT_REPLAN_SIGNALS_ENV.split(",") if s.strip()] or (
    "获取失败", "未取到", "未获取到", "检索失败", "搜索失败", "读取失败", "写入失败",
    "rate limit", "rate_limit", "ratelimit", "timeout", "超时",
    "traceback", "exception occurred", "api error", "tool call failed", "工具调用失败",
    "执行失败", "运行失败", "无法访问", "connection error", "连接失败", "access denied",
    "no data", "无数据", "empty result", "空结果",
)
# 观察窗口：_replan_evidence 扫描最近 N 条 observations。
# 原固定 6 条太窄 —— 实测错误信号后追加 ≥6 条消息即漏判，而一步 ReAct（recursion_limit 15）
# 必然产生十几条消息，上一步的真实信号经常被挤出窗口。默认 20 覆盖一整步；设 0 表示全量扫描。
_REPLAN_OBS_WINDOW = int(os.environ.get("AGENT_REPLAN_OBS_WINDOW", "20") or 0)
# 产物契约校验（第 3 类证据）：查最近 N 个落盘产物是否存在/非空/JSON 可解析且非空。
# 这是"数据驱动"的关键补丁 —— 此前只有失败关键词与文件存在性两类证据，
# 于是"文件写了但内容为空/不是约定结构"这种静默降级完全不会被发现。
_REPLAN_ARTIFACT_CHECK = os.environ.get("AGENT_REPLAN_ARTIFACT_CHECK", "1").lower() in ("1", "true", "yes")
_REPLAN_ARTIFACT_MAX = int(os.environ.get("AGENT_REPLAN_ARTIFACT_MAX", "3") or 3)


def _mark_progress(plan: list, current: int) -> list:
    """把已完成步（index < current）标记为 completed；其余保持 pending。

    独立于 LLM：即使再规划失败（后端不可达/解析异常），plan 的 status 也必须反映真实进度，
    否则 langgraph dev 里看到的永远是初始快照，"plan 没更新"无法归因。

    已标记 failed 的步**保持 failed 不被覆盖**：失败是终态事实，索引推进只代表"调度器走过了这一步"，
    不代表它成功了；若在此洗成 completed，UI/日志会把失败步误报为成功，掩盖真实问题。
    """
    new_plan = _normalize_plan(plan)
    for i in range(min(current, len(new_plan))):
        if new_plan[i].get("status") != "failed":
            new_plan[i]["status"] = "completed"
    return new_plan


def _mark_failed(plan: list, idx: int) -> list:
    """把第 idx 步（0-based）标记为 failed，其余步状态原样保留。

    用于子 agent 重试耗尽/异常退出时：supervisor 在派发时已把 current_step +1，若不显式标 failed，
    下一轮 _mark_progress 会按索引把它算成 completed，导致失败步被当成成功。
    """
    new_plan = _normalize_plan(plan)
    if 0 <= idx < len(new_plan):
        new_plan[idx]["status"] = "failed"
    return new_plan


def _plan_view(plan: list, current: int = -1) -> str:
    """把 plan 渲染成可读文本，用于日志/喂给 LLM。"""
    return "\n".join(
        f"{i + 1}. {'>> ' if i == current else '   '}"
        f"[{s.get('status', 'pending') if isinstance(s, dict) else 'pending'}] "
        f"{s.get('title', '') if isinstance(s, dict) else ''}: "
        f"{s.get('description', '') if isinstance(s, dict) else str(s)}"
        for i, s in enumerate(plan)
    )


def _should_replan(state: dict, plan: list, current: int) -> tuple:
    """判断这一轮是否值得为"再规划"花一次 LLM 调用。返回 (should: bool, reason: str)。

    背景（线上实证）：多轮再规划 BEFORE 与 AFTER 完全相同 —— 纯回显，LLM 开销白花。
    以下情形再规划在定义上不可能产出任何改变，直接跳过：
      1) current == 0：还没有任何一步执行过，没有新信息可供"根据执行情况改写计划"；
      2) 连续 _REPLAN_NOOP_MAX 次 BEFORE==AFTER 且**本轮也没有任何新证据**：说明模型对这份计划
         没有改写意愿，后续大概率继续空转 → 停用。
    例外（必须保留调用，否则会掩盖问题）：
      - 存在 failed 步：需要模型决定重试/改写/放弃，是再规划最有价值的场景；
      - 最后一步（current == len(plan)-1）：没有"剩余步骤"可改写，但"这一步还需不需要跑"
        正是早停（#6）要模型拍板的决策，价值高，不能省；
      - **本轮有证据（失败信号 / 文件断裂 / 产物违约）**：空转停用不得压掉真证据 ——
        否则一旦前两轮空转，后面即使 sub agent 交回空数据也永远不会被审视（省钱的门反而成了漏判的门）。
    """
    if _AGENT_REPLAN_DISABLE:
        return False, "AGENT_REPLAN_DISABLE=1（再规划已整体停用）"
    if current <= 0:
        return False, "首步派发：尚无任何执行结果，无新信息可供再规划"
    norm = _normalize_plan(plan)
    if any(s.get("status") == "failed" for s in norm):
        return True, "存在 failed 步，需要模型重新决策（重试/改写/放弃）"
    if current >= len(norm) - 1:
        return True, "最后一步：剩余步骤为空，但'是否仍需执行本步'由模型早停决策"
    streak = 0
    try:
        streak = int(state.get("replan_noop_streak") or 0)
    except (TypeError, ValueError):
        streak = 0
    # 证据门（#4）：先判证据，再判空转停用 —— 证据优先级高于空转停用。
    # 旧顺序（先 noop 后 evidence）会让"连续两次空转"一票否决掉后续所有真实问题，
    # 实测场景 F 即因此漏判：明明 sub agent 交回的是空数据，却因为 streak=2 而不再审视。
    ev, why = _replan_evidence(state, norm, current)
    if ev:
        if _REPLAN_NOOP_MAX > 0 and streak >= _REPLAN_NOOP_MAX:
            return True, f"{why}（本轮有证据，优先于连续 {streak} 次空转的停用判定）"
        return True, why
    if _REPLAN_NOOP_MAX > 0 and streak >= _REPLAN_NOOP_MAX:
        return False, (f"连续 {streak} 次再规划均为空转（BEFORE==AFTER）且本轮无新证据，"
                       f"已停用后续再规划以省 LLM 开销")
    return False, "无证据表明计划需改写（失败步/观察冲突/文件断裂/产物违约均无），跳过再规划以省 LLM"


def _artifact_contract_breach(state: dict) -> tuple:
    """第 3 类证据：已落盘产物是否满足最低契约。返回 (breached, reason)。

    为什么必须加这一类：前两类证据只认"失败关键词"和"文件存在性"，于是 sub agent
    "写了文件但内容是空的 / 不是约定的 JSON / JSON 是空数组"这种**静默降级**永远查不出来
    —— supervisor 看到文件在盘上就判定 handoff 成功，原样把"分析结构化数据"派给下游，
    整条链路带着坏数据跑到底，plan 一字不改。

    只做**确定性**校验（不调 LLM、不猜语义）：
      1) 文件不存在（产物清单里记了却没落盘）；
      2) 文件 size == 0；
      3) .json 无法解析，或解析出来是空 list / 空 dict。
    其余类型（md/png/csv…）只查存在性与空文件，不做内容断言，避免误判。
    只看最近 _REPLAN_ARTIFACT_MAX 个产物，避免扫全量历史。异常一律吞掉：
    证据采集绝不能反过来让再规划崩掉。
    """
    if not _REPLAN_ARTIFACT_CHECK:
        return False, ""
    try:
        arts = [a for a in (state.get("artifacts") or []) if isinstance(a, str) and a.strip()]
        for p in reversed(arts[-_REPLAN_ARTIFACT_MAX:]):
            try:
                if not os.path.exists(p):
                    return True, f"产物文件不在盘上: {p}"
                if os.path.getsize(p) == 0:
                    return True, f"产物文件为空(0 字节): {p}"
                if p.lower().endswith(".json"):
                    try:
                        with open(p, "r", encoding="utf-8", errors="ignore") as f:
                            data = json.load(f)
                    except (json.JSONDecodeError, ValueError) as je:
                        return True, f"JSON 产物无法解析: {p} ({type(je).__name__})"
                    if isinstance(data, (list, dict)) and len(data) == 0:
                        return True, f"JSON 产物为空容器: {p}"
            except OSError:
                continue
    except Exception as e:
        logger.warning(f"[replan] artifact contract check skipped ({e})")
    return False, ""


def _replan_evidence(state: dict, plan: list, current: int) -> tuple:
    """判断"计划需要改写"的廉价证据（不调 LLM）。返回 (has_evidence, reason)。

    对应 #4「有证据才再规划」：失败步/观察与计划冲突才调 LLM，做到单步零浪费。
    证据类型：
      1) 最近观察含强失败/中断信号（关键词扫描，见 _REPLAN_EVIDENCE_SIGNALS，可经
         AGENT_REPLAN_SIGNALS 覆盖）——上游步可能没拿到预期数据，剩余步前提或失效；
      2) 某个 pending 步的 description 引用了不存在的文件（handoff 断裂）——下游将无米下锅；
      3) 已落盘产物违反最低契约（不存在 / 空文件 / JSON 坏或空）——见 _artifact_contract_breach。
    三者皆无 → 返回 (False, "")，supervisor 跳过本轮再规划。
    （失败步本身由 _should_replan 单独判 True，不在此重复处理。）
    """
    # 1) 最近观察的失败/中断信号（窗口可配 _REPLAN_OBS_WINDOW，默认覆盖一整步的消息量）
    obs = state.get("observations") or []
    win = obs[-_REPLAN_OBS_WINDOW:] if _REPLAN_OBS_WINDOW > 0 else obs
    buf = []
    for m in win:
        if isinstance(m, (AIMessage, ToolMessage)):
            buf.append(_msg_text(m))
    low = "\n".join(buf).lower()
    for s in _REPLAN_EVIDENCE_SIGNALS:
        if s.lower() in low:
            return True, f"最近观察含失败/中断信号: '{s}'"
    # 2) pending 步引用了不存在的文件（handoff 断裂）
    arts = set(state.get("artifacts") or [])
    for step in plan[current:]:
        desc = f"{step.get('title', '')} {step.get('description', '')}"
        for tok in re.findall(r'[A-Za-z]:\\[^\s"]+|/[\w./\-]+\.(?:json|md|csv|png|txt|py)', desc):
            if tok not in arts and not os.path.exists(tok):
                return True, f"pending 步引用了不存在的文件: {tok}"
    # 3) 产物契约：文件在盘上但内容不可用（此前完全查不出来的一类静默降级）
    breached, why = _artifact_contract_breach(state)
    if breached:
        return True, f"产物违反最低契约: {why}"
    return False, ""


def _replan_tail(state: dict, plan: list, current: int, summary_ctx: str | None = None):
    """调用 LLM 审视并改写剩余步骤。返回 (current_agent, new_plan, finish_reason|None)。

    借鉴 single-agent demo 的 update_planner：
    - plan 为结构化步骤（含 status），只重写"未完成"的尾部（index > current），
      已完成步（index < current）保留并标记为 completed；
    - plan_goal 永不改变（对应 demo "don't change the goal"）；
    - 再规划上下文优先用跨步语义摘要 summary_ctx（_summarize_observations 提炼的要点清单，
      比原始 observations 更准更省 token）；摘要缺失时回落 _build_replan_context（原始 observations）。

    finish_reason 非 None 表示**模型主动早停**（#6）：模型返回 next=FINISH 且把 execution_plan
    显式截断到 <= current 步（即放弃剩余步骤）。只喊 FINISH 却不截断的一律驳回 —— 历史上模型
    "偷懒式 FINISH" 会让任务半途而废，故早停必须是可验证的显式动作，而不是一句话。

    安全约束（防死循环）：
    - 已完成步 plan[:current] 永不改写，只标 completed；
    - 若模型返回的 plan 更长，截断到原长（只减不增），杜绝无限追加；
    - 模型未返回 execution_plan 时**保留原尾部**（曾错误地把尾部丢掉导致提前 FINISH）；
    - current_step 由调用方负责 +1，本函数不回退。
    任何解析/调用异常都向上抛，由调用方回落"按原计划下一步"。
    """
    goal = state.get("plan_goal") or _goal_text(state)
    if summary_ctx is None and not _AGENT_SUMMARY_DISABLE:
        summary_ctx = _summarize_observations(state)
    elif summary_ctx is None:
        summary_ctx = ""
    ctx = summary_ctx or _build_replan_context(state)
    sys_prompt = supervisor_system_prompt.replace("{members}", ", ".join(members)) + "\n\n" + _date_context_str()
    sys_msg = SystemMessage(content=sys_prompt)
    before_view = _plan_view(_mark_progress(plan, current), current)
    is_last = current >= len(plan) - 1
    logger.info(f"supervisor re-planning step {current + 1}/{len(plan)}; plan BEFORE:\n{before_view}")
    # 剩余步骤为空时不再要求模型回显整份计划（这正是"纯回显"浪费的大头）：只让它做"跑还是不跑"的决策。
    revision_rule = (
        "There are NO remaining steps to revise. Decide ONLY whether this last step still needs to run:\n"
        "  - If the completed steps have ALREADY fully achieved the goal → you may return next=\"FINISH\" "
        "ALONE (no 'execution_plan' needed) to drop this redundant final step; OR return next=\"FINISH\" with "
        "an 'execution_plan' containing ONLY the first {n} completed steps (i.e. DROP this step).\n"
        "  - Otherwise return the agent for this step and OMIT 'execution_plan' entirely.\n"
        "Do NOT echo the plan back."
    ).format(n=current) if is_last else (
        "Decide the agent for the CURRENT step, then revise ONLY the REMAINING steps (index > current) "
        "based on what actually happened. You may skip/merge/rewrite remaining steps, but you MUST: "
        "1) keep the goal unchanged; 2) NOT increase total plan length; 3) NOT re-run completed steps; "
        "4) keep each remaining step's 'status' as 'pending' (completed steps are already marked).\n"
        # 关键补充：以前模型只知道"某步 failed"，改写出来的一句话常常是"重新执行：…"，
        # 既没说清根因也没给出任何新约束 → 下游必然再撞同一个坑（如再撞一次 ReAct 步数上限）。
        # 现要求把"如何避免重蹈覆辙"显式写进步骤 description，成为下游能直接执行的约束。
        "5) RE-DOING A FAILED STEP: if you keep or re-do a step that previously failed, its 'description' "
        "MUST encode the concrete constraints that avoid repeating that failure (from the failure notes "
        "below) — never write a bare 'redo the same thing'.\n"
        "If the remaining steps need NO change, OMIT 'execution_plan' entirely — do NOT echo it back.\n"
        "EARLY FINISH: if the completed steps have already fully achieved the goal and NO remaining step "
        "is needed, return next=\"FINISH\" AND an 'execution_plan' containing ONLY the first {n} steps "
        "(i.e. drop every step from index {n} onward). A FINISH that does not truncate 'execution_plan' "
        "will be REJECTED and the step will run anyway."
    ).format(n=current)
    # 失败回溯区：把"本步/前序失败步为什么失败 + 已给过哪些改进要求"一并喂给再规划模型。
    # 在此之前 prompt 里唯一的失败信息是 [failed] 状态本身，模型只能猜着写"重新执行：…"，
    # 于是同一根因被连续踩三次（2026-10-02 step3/4/5 全撞 ReAct 步数上限）。
    # supervisor 常把失败步合并/改写成本步，故这里同样用"本步 + 前 N 步"的回溯窗口（见 handoff）。
    fail_ctx = _render_step_failures(state, current)
    user_msg = HumanMessage(content=(
        f"User goal (NEVER change this):\n{goal}\n\n"
        f"Current execution_plan ('>>' marks the step being dispatched NOW, index {current}):\n{before_view}\n\n"
        f"Summary of completed steps (key points — full data is in the listed files):\n{ctx}\n\n"
        + (f"{fail_ctx}\n\n" if fail_ctx else "")
        + f"{revision_rule}\n\n"
        "Return strict JSON with 'next' (agent for current step or FINISH) and 'reason'."
    ))
    messages = [sys_msg, user_msg]
    # 结构化输出 + parse 失败带错误重试（鲁棒性#3）；全败才回落裸调用+手动抽 JSON。
    parsed = _structured_with_retry(supervisor_llm, messages, Router, label="re-plan")
    if not isinstance(parsed, dict):
        ai = supervisor_llm.invoke(messages)
        content = ai.content if isinstance(ai, AIMessage) else str(ai)
        parsed = _extract_json_obj(content)
    if not isinstance(parsed, dict):
        raise ValueError("re-plan produced no parseable JSON")

    # 当前步 agent：校验模型给的 next，失败回落解析 plan[current]
    next_agent = parsed.get("next")
    valid = [m.replace("_agent", "") for m in members] + ["FINISH"]
    if not (isinstance(next_agent, str) and next_agent.replace("_agent", "") in valid):
        next_agent = _parse_target_agent(plan[current])

    # 基线：已完成步标 completed，未完成步沿用原计划（模型不给新计划时这就是最终结果）。
    new_plan = _mark_progress(plan, current)
    revised = parsed.get("execution_plan")
    finish_reason = None
    norm_rev = _normalize_plan(revised) if isinstance(revised, list) and revised else None

    if next_agent == "FINISH":
        # 早停必须"显式截断计划"才被采信：execution_plan 长度 <= current 表明模型真的放弃了剩余步骤。
        # 只喊 FINISH 却不截断 → 一般驳回（防偷懒式早停导致任务半途而废）。
        if current > 0 and norm_rev is not None and 0 < len(norm_rev) <= current:
            # 原有早停通道：模型显式截断计划
            new_plan = _mark_progress(norm_rev, len(norm_rev))
            finish_reason = str(parsed.get("reason") or "").strip() or "model judged the goal already achieved"
            logger.info(
                f"supervisor EARLY FINISH accepted after {current}/{len(plan)} steps; "
                f"remaining {len(plan) - len(norm_rev)} step(s) dropped. reason={finish_reason}"
            )
        elif current >= 1 and is_last:
            # B：最后一步裸 FINISH 直接采信为早停。最后一步之后没有剩余步，丢弃它即等价于
            # "把计划截断到 <= current"，无需模型手写截断。用于"最后一步（如发邮件/再总结）多余"的场景。
            kept = plan[:current] if current > 0 else []
            new_plan = _mark_progress(kept, len(kept)) if kept else plan
            finish_reason = str(parsed.get("reason") or "").strip() or \
                "model judged goal already met at last step (bare FINISH accepted)"
            logger.info(
                f"supervisor EARLY FINISH accepted (last-step bare FINISH): "
                f"dropped final step {current + 1}/{len(plan)}. reason={finish_reason}"
            )
        else:
            logger.warning(
                "supervisor FINISH rejected: model did not truncate 'execution_plan' "
                f"(next=FINISH at step {current + 1}/{len(plan)}); falling back to executing current step."
            )
            next_agent = _parse_target_agent(plan[current])
    elif norm_rev is not None and len(norm_rev) > current:
        tail = norm_rev[current:]  # 模型掌控 current 及之后
        if tail:
            new_plan = new_plan[:current] + tail
            if len(new_plan) > len(plan):          # 只减不增：截断到原长
                new_plan = new_plan[:len(plan)]
    # 其余情形（模型未给计划 / 给了比 current 更短的计划却仍要跑 agent）→ 保留原尾部，避免出现
    # "plan 比 current_step 还短"导致 supervisor 索引越界。
    if len(new_plan) < current:            # 安全兜底
        new_plan = _mark_progress(plan, current)
    logger.info(f"supervisor re-plan result: agent={next_agent}; plan AFTER:\n{_plan_view(new_plan)}")
    return next_agent, new_plan, finish_reason
