"""子 Agent 工厂 + 节点工厂。

- create_resilient_agent(llm, tools, system_prompt, ...)：标准化 agent（包 ToolCallLoggingMiddleware）；
- create_agents()：构造 6 个语义角色 agent，返回 {node_name: agent}；
- create_resilient_node(agent)：把一个 agent 包装成带错误恢复 / 重试 / 快照回滚 / 日期上下文注入 /
  上下文压缩 / 产物落盘的 LangGraph 节点闭包；
- create_nodes()：基于 create_agents() 批量产出节点字典 {node_name: node}。

原定义位于 agent.py:417-488（agent 工厂 + 实例）+ 1367-1548（节点工厂 + 实例），拆分时整体迁入。
依赖：llm / prompt / tools / context / compress / handoff / summary / plan / runlog。
"""
import os
import re
import json
import logging
import concurrent.futures.thread
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.errors import GraphRecursionError, GraphBubbleUp
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

from llm import (
    ToolCallLoggingMiddleware, CustomContextMiddleware, concurrency_guard_middleware,
    chat_llm, db_llm, coder_llm, crawler_llm, rag_llm, context_engineer_llm,
    supervisor_llm,
)
from prompt import (
    db_system_prompt, supervisor_system_prompt, rag_system_prompt,
    agentic_context_system_prompt, crawler_system_prompt, coder_system_prompt, chat_system_prompt,
)
from tools import (
    read_file, create_file, str_replace, send_qq_email,
    add_sale, delete_sale, update_sale, query_sales, query_table_schema, execute_sql,
    python_repl, shell_exec,
    get_nasdaq_top_gainers, get_crypto_sentiment_indicators, resilient_tavily_search,
    get_current_time,
    list_files_metadata, grep_files,
    save_context_snapshot, list_context_snapshots, evaluate_output, restore_snapshot,
    _run_tool,
)
from context import _date_context_str, _data_freshness_check
from compress import _compress_messages, _msg_text
from handoff import _step_assignment_text, _save_artifact, TMP_DIR
from summary import _summarize_observations, _AGENT_SUMMARY_DISABLE, _obs_total
from plan import _mark_failed
from planutil import _extract_json_obj
from hitl import hitl_middleware, HITL_ENABLED
from runlog import set_current, ensure_run, log_event, run_started_at

logger = logging.getLogger(__name__)

# === 鲁棒性#3：单子 agent 内部 ReAct 循环的显式步数上限（recursion_limit）===
# 不再吃 LangGraph 默认 25；超限抛 GraphRecursionError，已被 node 捕获并标 failed，杜绝无限工具循环烧钱。
_AGENT_MAX_ITER = int(os.environ.get("AGENT_MAX_ITERATIONS", "15"))

# === 鲁棒性#1：Critic 质量门（completed ≠ 做对了）===
# 设计原则「宁可漏判，绝不冤枉」：只在**高置信度失败**时才拒，最大限度避免把正确产出误判为不合格
# （误判会触发无谓的带反馈重做、浪费 token，甚至把正确步标 failed）。规则层零 LLM 成本、默认开；
# 语义层（LLM 判定产出是否真满足本步）默认关（AGENT_CRITIC_LLM=1 开启），避免每步多一次 LLM 拖慢。
# min_len 只是"空泛非答复"的下限（默认极低），不是质量门槛——真实答复再短也不会被它拦下。
_CRITIC_MIN_LEN = int(os.environ.get("AGENT_CRITIC_MIN_LEN", "4"))
_CRITIC_RETRIES = int(os.environ.get("AGENT_CRITIC_RETRIES", "2"))  # 质量门不合格时「带反馈重做同一步」的最大次数
_CRITIC_LLM = os.environ.get("AGENT_CRITIC_LLM", "0").lower() in ("1", "true", "yes")
# 交付型步识别：只认明确的「产出/落盘」动词。绝不含"读取/分析/总结/查询/文件/报告"等只读意图词，
# 以免把"读取文件并总结"这类步误判为交付步、进而因"没给出落盘路径"而冤枉它。
_DELIVERABLE_VERBS = (
    "生成", "保存", "写入", "落盘", "导出", "创建", "产出", "存为", "存到", "写成",
    "整理成", "整理为", "汇总成", "编写", "撰写", "输出到", "记录到", "做成", "画成", "绘制",
    "generate", "export", "persist", "compile", "save as", "save to", "save the",
    "write to", "write into", "output to", "create a", "create the", "create file",
    "produce a", "produce the",
)
# 落盘路径识别（尽量宽松，降低「产出正确却被误判」的概率）：接受绝对/相对/裸文件名，覆盖常见交付扩展名。
_PATH_EXTS = r"(?:md|markdown|json|csv|tsv|png|jpg|jpeg|svg|py|txt|xlsx|xls|html|htm|pdf|docx|log)"
_PATH_RE = rf"regex:[\w.:/\\\-]+\.{_PATH_EXTS}"


def _is_deliverable_step(step) -> bool:
    """本步是否要求「产出/落盘一个文件」——只有这类步才强制校验落盘路径。

    只匹配明确的写动词：宁可对措辞特殊的交付步漏检（false negative，顶多不强制路径，无害），
    也绝不把"读取/分析/总结/查询"步误判成交付步而冤枉它（false positive，会触发无谓重做/误标 failed）。
    """
    desc = (f"{step.get('title', '')} {step.get('description', '')}".lower()
            if isinstance(step, dict) else str(step).lower())
    return any(k in desc for k in _DELIVERABLE_VERBS)


def _critic_criteria(step) -> str:
    """由本步 title/description 自动推导 Critic 规则（供日志/反馈展示；实际判定见 _critic_gate）。"""
    if _is_deliverable_step(step):
        return "交付型步：非空，且最终回复或工具输出中必须出现落盘文件路径"
    return f"非交付步：非空且不少于 {_CRITIC_MIN_LEN} 字（仅挡空泛非答复，非质量门槛）"


def _critic_feedback_text(reason: str, criteria: str, attempt: int, budget: int) -> str:
    """把 Critic 不合格原因翻译成 sub agent 能据以改进的明确指令（避免第二轮又生成同样的错）。"""
    low = (reason or "").lower()
    if "empty" in low or "min_len" in low:
        hint = ("你的产出为空或过短。请补全完整内容与关键结论（不要只回一句「已完成/见文件」），"
                "把下游真正需要的数据/结论直接写出来。")
    elif "路径" in (reason or "") or "regex" in low or "path" in low:
        hint = ("本步被判定为交付型任务，但你最终回复与本步工具输出里都没有出现有效的落盘文件路径。"
                "请确认文件确实已写入磁盘，并在回复中明确给出该文件路径（绝对或 tmp\\ 相对路径均可）；"
                "若并未真正落盘，请说明原因并补做。")
    else:
        hint = "请针对上述不合格原因逐条修正后重新产出，不要重复同样的问题。"
    return (
        f"[Critic 质量门 · 第 {attempt}/{budget} 次带反馈重做] 你上一轮的产出未通过验收。\n"
        f"不合格原因：{reason}\n"
        f"改进要求：{hint}\n"
        f"验收标准：{criteria}\n"
        "若确实无法满足验收标准，请明确说明缺失了什么、为什么无法完成（不要用空泛的确认糊弄）。"
    )


def _critic_gate(state: dict, step, final_text: str, agent_name: str, tool_evidence: str = "") -> tuple:
    """Critic 质量门：返回 (passed, reason)。「宁可漏判，绝不冤枉」——只在高置信度失败时才拒。

    硬拒只有两种（都无歧义、几乎不可能冤枉）：
      A) 产出完全为空（final_text 与本步工具输出 tool_evidence 都空）——agent 明显没干活；
      B) 交付型步（明确要求生成/保存文件）却在「最终回复 + 本步工具输出」里都找不到任何落盘路径
         ——声称交付却无落盘证据。
    其余一律放行：
      - 交付型步只要找到路径 → 立即通过（不再用长度卡它：文件已生成，回复短不算失败，避免冤枉）；
      - 非交付步只要非空、且不是"一两个字的空泛答复"（min_len 下限，默认极低）→ 通过。
    不通过 → 带反馈重做同一步（≤ _CRITIC_RETRIES 次），仍不合格才标 failed 交 supervisor 再规划。
    评估器自身异常时 fail-open（绝不因工具报错而阻断/冤枉主流程），仅告警。
    """
    try:
        text = final_text or ""
        ev = tool_evidence or ""
        # A) 完全空产出（唯一无歧义的"没干活"）
        if not text.strip() and not ev.strip():
            return False, "rule gate: Output is empty but expected non-empty."
        if _is_deliverable_step(step):
            # B) 交付型步：找到落盘路径即通过（文本或工具输出任一处）；找不到才拒
            pres = _run_tool(evaluate_output, _PATH_RE, f"{text}\n{ev}")
            if isinstance(pres, dict) and pres.get("passed") is True:
                return True, ""      # 交付物客观存在 → 放行，不因回复短而冤枉
            return False, "rule gate: 交付型步未给出落盘文件路径（最终回复与工具输出均未发现有效路径）"
        # 非交付步：只挡"空泛到不可能是真实答复"的极短产出（默认下限极低，几乎不冤枉）
        if text.strip():
            base = _run_tool(evaluate_output, f"not empty;min_len:{_CRITIC_MIN_LEN}", text)
            if isinstance(base, dict) and base.get("passed") is False:
                return False, f"rule gate: {base.get('reason', '')}"
    except Exception as ce:
        logger.warning(f"[critic] rule eval error ({ce}); fail-open")
        return True, "critic rule eval skipped"
    if _CRITIC_LLM:
        desc = (f"{step.get('title', '')} {step.get('description', '')}"
                if isinstance(step, dict) else str(step))
        try:
            ai = supervisor_llm.invoke([
                SystemMessage(content=(
                    "You are a strict Critic. Judge ONLY whether the DELIVERABLE satisfies the STEP "
                    "REQUIREMENT. Reply with a single JSON object {\"passed\": true|false, "
                    "\"reason\": \"<short>\"}. Be concise; do not invent requirements."
                )),
                HumanMessage(content=f"STEP REQUIREMENT:\n{desc}\n\nDELIVERABLE:\n{final_text[:4000]}"),
            ])
            verdict = _extract_json_obj(ai.content if isinstance(ai, AIMessage) else str(ai)) or {}
            if verdict.get("passed") is False:
                return False, f"semantic gate: {verdict.get('reason', '')}"
        except Exception as ce:
            logger.warning(f"[critic] semantic eval error ({ce}); fail-open")
    return True, ""


def _is_interpreter_shutdown(exc: Exception) -> bool:
    """判断异常是否由"解释器 / 线程池正在关闭"引起。

    langgraph dev 热重载或 Ctrl+C 杀进程时，同步节点所在的 ThreadPoolExecutor 会拒绝新任务
    （RuntimeError: cannot schedule new futures after interpreter shutdown / after shutdown）。
    这属于外部进程击杀而非业务失败：重试必然继续失败，也不应把该步标 failed。
    """
    msg = str(exc)
    return "cannot schedule new futures" in msg or bool(concurrent.futures.thread._shutdown)


# === 创建 Agent（使用 LangChain 1.4.x create_agent + Middleware for Context Engineer）===
def create_resilient_agent(llm, tools, system_prompt, agent_name="Agent", middleware=None):
    """创建标准化 Agent with resilience"""
    system_msg = SystemMessagePromptTemplate.from_template(system_prompt)
    prompt = ChatPromptTemplate.from_messages([system_msg, MessagesPlaceholder(variable_name="messages")])
    # 工具调用日志中间件置于最外层（first defined = outermost），保证它包住其余中间件里的工具调用；
    # 传入 agent_name，让每条 [tool] 日志带上归属，避免控制台里跨 agent 误读。
    # HITL 中间件（默认关；开启后仅对 danger 级工具生效）置于末尾：在 after_model 阶段对危险
    # 工具调用触发 interrupt()，交由 Studio 人工 approve/edit/reject 后才真正执行。
    tool_names = [getattr(t, "name", None) for t in (tools or [])]
    # 中间件顺序（first defined = outermost）：日志 → 并行安全护栏（有副作用工具串行）→ 自定义 → HITL。
    # 护栏仅在 AGENT_PARALLEL_TOOL_CALLS 开启时非空；关闭时为 []，零开销、行为与改造前一致。
    mw = ([ToolCallLoggingMiddleware(agent_name)] + concurrency_guard_middleware()
          + list(middleware or []) + hitl_middleware(tool_names))
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,  # Passed directly in 1.4.x
        middleware=mw,
        name=agent_name
    )


def create_agents() -> dict:
    """构造 6 个语义角色 agent，返回 {节点名: agent}。"""
    # 1. Chat Agent
    chat_agent = create_resilient_agent(
        chat_llm,
        tools=[read_file, grep_files, create_file, str_replace, send_qq_email, get_current_time],
        system_prompt=chat_system_prompt,
        agent_name="ChatAgent"
    )

    # 2. DB Agent
    db_agent = create_resilient_agent(
        db_llm,
        tools=[add_sale, delete_sale, update_sale, query_sales, query_table_schema, execute_sql, get_current_time],
        system_prompt=db_system_prompt,
        agent_name="DBAgent"
    )

    # 3. Code Agent
    # 工具隔离：取数（联网检索）是 crawler_agent 的专职。code_agent 若握着搜索工具，在只看到用户原始
    # query 时会"顺手再搜一遍"（本轮线上现象：crawler 搜完，code/chat 又用同一 query 重复检索）。
    # code_agent 的输入应当是上游落盘的产物文件，不是它自己去抓的新数据。
    code_agent = create_resilient_agent(
        coder_llm,
        tools=[python_repl, create_file, read_file, grep_files, str_replace, shell_exec, get_current_time],
        system_prompt=coder_system_prompt,
        agent_name="CodeAgent"
    )

    # 4. Crawler Agent
    crawler_agent = create_resilient_agent(
        crawler_llm,
        tools=[get_nasdaq_top_gainers, get_crypto_sentiment_indicators, resilient_tavily_search, create_file, get_current_time],
        system_prompt=crawler_system_prompt,
        agent_name="CrawlerAgent"
    )

    # 5. RAG Agent
    rag_agent = create_resilient_agent(
        rag_llm,
        tools=[list_files_metadata, read_file, grep_files, get_current_time],
        system_prompt=rag_system_prompt.format(file_path=os.getcwd() + "\\documents"),
        agent_name="RAGAgent"
    )

    # 6. Context Engineer (with Custom Middleware)
    context_engineer = create_resilient_agent(
        context_engineer_llm,
        tools=[save_context_snapshot, list_context_snapshots, evaluate_output, restore_snapshot, get_current_time],
        system_prompt=agentic_context_system_prompt,
        agent_name="ContextEngineer",
        middleware=[CustomContextMiddleware(), SummarizationMiddleware(model=context_engineer_llm, trigger=('tokens', 1000))]
    )

    return {
        "chat_agent": chat_agent,
        "db_agent": db_agent,
        "code_agent": code_agent,
        "crawler_agent": crawler_agent,
        "rag_agent": rag_agent,
        "context_engineer_agent": context_engineer,
    }


# === 通用 Agent 节点（带错误恢复，集成 Middleware行为）===
def create_resilient_node(agent):
    """创建带错误恢复的节点函数"""
    def node(state: dict, config=None) -> dict:
        # 所有节点都能看到当前计划，增强可观测性（execution_plan 可能为 None，统一当成空列表）
        _plan = state.get("execution_plan") or []
        _cur = state.get("current_step", 0)
        _step = _plan[_cur - 1] if (_plan and 0 < _cur <= len(_plan)) else None
        if isinstance(_step, dict):
            _step_txt = f"{_step.get('title', '')} | {_step.get('description', '')}"
        else:
            _step_txt = str(_step) if _step is not None else "No plan"
        _mk = state.get("memory_key")
        # 幂等守卫：计划里该步已标 completed → 说明上一次运行（或续跑）已成功产出，
        # 直接跳过本轮执行，避免续跑 / 重复派发时重复调 LLM、重复落盘产物（# 用户加固需求）。
        # 正常流不会误触发：supervisor 派发第 current 步时只对 index<current 标 completed，
        # 当前步本身在节点进入时仍是 pending（见 plan._mark_progress）。
        if _plan and 0 < _cur <= len(_plan) and isinstance(_plan[_cur - 1], dict) \
                and _plan[_cur - 1].get("status") == "completed":
            logger.info(
                f"[idempotent] step {_cur}/{len(_plan)} already 'completed' → skip re-execution "
                f"(resume / double-dispatch safety)."
            )
            log_event(
                f"[idempotent] step {_cur}/{len(_plan)} already 'completed' → skip re-run "
                f"(no new artifact). step={_step_txt}",
                _mk,
            )
            return {}
        # 产物感知的幂等守卫（二道防线）：_save_artifact 在节点 return（checkpoint 提交）**之前**落盘，
        # 若进程恰好死在这个窗口（如 langgraph dev 热重载击杀），续跑时该步在计划里仍是 pending，
        # 上面基于 status 的守卫拦不住 → 整步被完整重做（2026-09-30 线上现象：step3 产物 11:44:40 已写出，
        # 11:44:49 续跑又把同样的 str_replace/脚本原样跑了一遍）。
        # 命中条件（两路）：
        #   a) state["artifacts"]（随 checkpoint 提交，天然属于本 run）里已有 __step{N}__ 产物；
        #   b) tmp/ 磁盘扫描：仅限 mtime 晚于**本次 run 启动时刻**的文件 —— tmp/ 永久累积，
        #      历史 run 的同号 step 产物（实测有 4 个 __step3__）绝不能误认。启动时刻优先读
        #      state["run_started_at"]（supervisor 首次规划时写入，随 checkpoint 跨进程续跑存活），
        #      兜底读本进程 runlog；两者都拿不到时宁可放弃 b) 重做一遍，也不冒跳过错误产物的风险。
        # 例外：该步已被显式标 failed（supervisor 再规划后重试同一步）时不跳过，否则失败步永远无法重做。
        if _plan and 0 < _cur <= len(_plan) and isinstance(_plan[_cur - 1], dict) \
                and _plan[_cur - 1].get("status") != "failed":
            _pat = re.compile(rf"__step{_cur}__")
            _hit = next((p for p in (state.get("artifacts") or []) if _pat.search(str(p))), None)
            if _hit is None:
                _start = None
                _rs = state.get("run_started_at")
                if _rs:
                    try:
                        _start = datetime.fromisoformat(str(_rs))
                    except ValueError:
                        _start = None
                if _start is None:
                    _start = run_started_at()  # 兜底：本进程内 runlog 记录的启动时刻（跨进程为 None）
                if _start is not None:
                    _hit = next(
                        (str(p) for p in TMP_DIR.glob(f"*__step{_cur}__*")
                         if p.is_file()
                         and datetime.fromtimestamp(p.stat().st_mtime) >= _start),
                        None,
                    )
            if _hit:
                logger.info(
                    f"[idempotent] step {_cur}/{len(_plan)} artifact already exists → skip re-execution: {_hit}"
                )
                log_event(
                    f"[idempotent] step {_cur}/{len(_plan)} artifact already on disk → skip re-run: {_hit}. step={_step_txt}",
                    _mk,
                )
                return {
                    "messages": [AIMessage(content=(
                        f"[idempotent] Step {_cur} artifact already produced ({os.path.basename(str(_hit))}); "
                        "skipped re-execution after interrupted run / resume."
                    ))],
                    "sender": agent.name,
                    "current_step": _cur,
                    "run_started_at": state.get("run_started_at"),
                }
        # 让本步内子 agent 触发的工具（如 tavily 检索）把日志落到正确的运行日志文件。
        set_current(_mk)
        ensure_run(_mk)
        logger.info(f"Executing plan step {_cur}/{len(_plan)}: {_step_txt}")
        max_retries = 3
        critic_retries = _CRITIC_RETRIES      # 质量门「带反馈重做同一步」的独立预算（不占用异常重试）
        attempt = 0                            # 异常重试计数
        critic_attempt = 0                     # 质量门重试计数
        critic_feedback = None                 # 上一轮不合格原因（喂回 sub agent 让它针对性改进）
        critic_prev_output = ""                # 上一轮被拒的产出（让 sub agent 看到自己写了啥）
        while attempt < max_retries:
            try:
                # 执行 Agent (LangChain 1.0 invoke)
                # 注入当前日期上下文：模型无实时时钟，必须显式告知"今天/昨天"才能正确解析
                # 相对时间（如"昨天"），否则会瞎猜年份。
                # 关键修复：create_agent 内部会把 system_prompt 作为 SystemMessage 置于消息列表最前面
                # (factory.py ~1451: messages = [request.system_message, *request.messages])。若此处再往
                # state["messages"] 塞一条 SystemMessage(date_ctx)，最终会冒出"两条 system 消息"，第二条
                # 落在非开头位置，触发 vLLM/OpenAI 的 400 错误 "System message must be at the beginning."。
                # 故把日期上下文前缀到"第一条 HumanMessage"（无则新建一条 HumanMessage），保证整条链路
                # 只有 create_agent 那一条 SystemMessage 且位于开头。
                # agent.invoke 只回传本轮"新生成"的消息（AIMessage + ToolMessage），不会回传输入的
                # HumanMessage，因此全局 state 不会被日期前缀污染，无需剥离。
                #
                # 本次改造（根治"下游 agent 各自重跑一遍"）：
                #  - 第一条 HumanMessage 前缀日期上下文；
                #  - 末尾追加一条 HumanMessage 承载"本步任务指令"（第 N/M 步 + title/description +
                #    上游产物路径 + 禁止重跑上游工作），让子 agent 明确知道自己这一步该产出什么；
                #  - 中间的历史消息先经 _compress_messages 做 pair-safe 压缩，抑制 context 膨胀。
                date_ctx = _date_context_str()
                run_state = dict(state)
                msgs = list(state.get("messages", []))
                prefixed = False
                for i, m in enumerate(msgs):
                    if isinstance(m, HumanMessage):
                        new_content = date_ctx + "\n\n" + m.content if isinstance(m.content, str) else m.content
                        msgs[i] = HumanMessage(content=new_content)
                        prefixed = True
                        break
                if not prefixed:
                    msgs = [HumanMessage(content=date_ctx)] + msgs
                compressed = _compress_messages(msgs)
                # 跨步语义摘要：让子 agent 拿到"上游已完成步的要点"，而不是只看被截断的原始历史
                # （或完全看不到上游做了什么）。摘要与文件 handoff 共同承载跨步信息，故原始历史可安全截断。
                summary_ctx = state.get("plan_summary") or ""
                summary_fresh = False
                if not summary_ctx and not _AGENT_SUMMARY_DISABLE:
                    summary_ctx = _summarize_observations(state)
                    summary_fresh = bool(summary_ctx)
                extra = []
                if summary_ctx:
                    extra.append(HumanMessage(content=(
                        "[Summary of completed upstream steps — treat as context; DO NOT redo this work, "
                        "full data is in the files listed]\n" + summary_ctx
                    )))
                assignment = _step_assignment_text(state, agent.name)
                run_state["messages"] = compressed + extra + [HumanMessage(content=assignment)]
                logger.info(f"{agent.name} step assignment:\n{assignment[:400]}")
                # 持久化本步任务下发（含第 N/M 步、title/description、上游产物路径、禁重跑规则），
                # 便于事后在 log/ 下核对 supervisor 到底给子 agent 派了什么活。
                log_event(f"{agent.name} step assignment (step {_cur}/{len(_plan)}):\n{assignment}", _mk)
                if critic_feedback:
                    # 带反馈重做：把上一轮被拒产出 + 不合格原因 + 改进要求追加进去，让同一个 sub agent
                    # 明确知道「上一轮为什么不合格、这次要怎么改」，避免它原样再产出一次同样的错误结果。
                    run_state["messages"] = run_state["messages"] + [
                        AIMessage(content=critic_prev_output or "(上一轮无有效产出)"),
                        HumanMessage(content=critic_feedback),
                    ]
                    log_event(f"{agent.name} critic-retry #{critic_attempt} feedback:\n{critic_feedback}", _mk)
                invoke_config = {"recursion_limit": _AGENT_MAX_ITER}
                if config is not None and HITL_ENABLED:
                    # 透传父 config：让子 agent 内的 interrupt()（HITL）能冒泡到外层图并被 Studio 恢复。
                    # 仅在 HITL 开启时透传——关闭时保持原有「子图无 checkpointer、每次全新执行」的行为，零回归。
                    invoke_config = {**config, "recursion_limit": _AGENT_MAX_ITER}
                result = agent.invoke(run_state, config=invoke_config)

                # 保存快照（每 3 轮对话一次，middleware handles visualization）
                if len(state["messages"]) % 3 == 0:
                    snap = _run_tool(save_context_snapshot,
                        name=f"node_{agent.name}",
                        content=json.dumps({
                            "messages": [getattr(m, "content", "") for m in state["messages"][-5:]],  # 最近5条
                            "sender": state.get("sender"),
                            "timestamp": datetime.now().isoformat(),
                        }, ensure_ascii=False),
                    )
                    snap_path = snap.get("path") if isinstance(snap, dict) else snap
                    snapshot_id = os.path.splitext(os.path.basename(snap_path))[0] if isinstance(snap_path, str) else snap_path
                    state["snapshot_id"] = snapshot_id
                    logger.info(f"Snapshot saved: {snapshot_id}")

                # 累积 observations：本轮新生成的 ToolMessage + 最终 AIMessage（对应 demo 的 state["observations"]）
                # 供 supervisor 再规划时作为"已完成步的真实产出"上下文，避免只看单条文本。
                prev_obs = list(state.get("observations", []) or [])
                new_obs = list(result.get("messages", [])) if isinstance(result, dict) else []
                observations = (prev_obs + new_obs)[-40:]  # 上限保留最近 40 条，防止无限增长
                # 单调累计计数：observations 到 40 条后从头部丢弃，len() 不再增长，
                # 增量摘要必须靠这个计数判断"本步是否有新产出"，否则会被误判为无新增而跳过摘要。
                obs_total = int(state.get("obs_total") or 0) + len(new_obs)

                # 产物落盘：把本步最终答复写到 tmp/ 并记入 artifacts，作为下游 handoff 通道。
                # 大块数据走文件而非 context，既避免 context 膨胀，也保证下游能拿到全量而非截断版。
                final_text = ""
                for m in reversed(new_obs):
                    if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
                        final_text = _msg_text(m)
                        break
                # 数据时效性校验：产出若带 AS_OF 标记且属"当日但已滞后数小时"，显式告警到运行日志。
                # 场景：22:54 盘后运行，crawler 抓到 10:04 的盘中快照，下游却当作当前行情研判今日走势，
                # 全链路无任何提示。此处做确定性校验（不依赖 LLM 自觉），并把告警喂给下游上下文。
                fresh_warn = _data_freshness_check(final_text)
                if fresh_warn:
                    logger.warning(fresh_warn)
                    log_event(f"[freshness] {agent.name}: {fresh_warn}", _mk)
                artifacts = list(state.get("artifacts") or [])
                art_path = _save_artifact(agent.name, state.get("current_step", 0), final_text)
                if art_path:
                    artifacts.append(art_path)

                # Critic 质量门（鲁棒性#1）：节点返回成功≠产出正确。规则层免费、语义层可选。
                # tool_evidence：本步工具输出（create_file/save 的成功回执等）——交付型步的落盘路径
                # 允许出现在这里而不只是最终回复里，降低「文件已建但回复没复述路径」的误判。
                tool_evidence = "\n".join(
                    _msg_text(m) for m in new_obs if isinstance(m, ToolMessage)
                )
                passed, reason = _critic_gate(state, _step, final_text, agent.name, tool_evidence=tool_evidence)
                if not passed:
                    if critic_attempt < critic_retries:
                        # 带反馈重做同一步（不消耗异常预算）：明确告诉 sub agent 上一轮为什么不合格、
                        # 这次要怎么改，否则它极可能原样再产出一次错误结果。
                        critic_attempt += 1
                        critic_prev_output = final_text
                        critic_feedback = _critic_feedback_text(
                            reason, _critic_criteria(_step), critic_attempt, critic_retries)
                        logger.warning(
                            f"[critic] step {_cur} REJECTED for {agent.name} "
                            f"(retry {critic_attempt}/{critic_retries}): {reason}")
                        log_event(
                            f"[critic] step {_cur}/{len(_plan)} rejected → retry {critic_attempt}/{critic_retries} "
                            f"with feedback: {reason}", _mk)
                        continue
                    # 反馈重做仍不合格 → 标 failed（不会被 _mark_progress 洗成 completed），交 supervisor 再规划
                    logger.warning(f"[critic] step {_cur} REJECTED for {agent.name} after "
                                   f"{critic_retries} feedback-retries: {reason}")
                    log_event(f"[critic] step {_cur}/{len(_plan)} deliverable rejected after retries → mark failed: {reason}", _mk)
                    return {
                        "messages": [AIMessage(content=(
                            f"[critic] Step {_cur} deliverable failed the quality gate after "
                            f"{critic_retries} feedback-retries: {reason}. Step marked FAILED for re-plan."
                        ))],
                        "sender": agent.name,
                        "execution_plan": _mark_failed(_plan, _cur - 1),
                        "current_step": _cur,
                        "run_started_at": state.get("run_started_at"),
                    }

                return {
                    "messages": result["messages"],
                    "sender": agent.name,
                    "error_count": 0,
                    "snapshot_id": state.get("snapshot_id"),
                    # 把计划状态原样带回，让 supervisor 能继续追踪进度（None 归一为空列表）
                    "execution_plan": state.get("execution_plan") or [],
                    "plan_goal": state.get("plan_goal"),
                    "run_started_at": state.get("run_started_at"),
                    "observations": observations,
                    "obs_total": obs_total,
                    "artifacts": artifacts,
                    "current_step": state.get("current_step", 0),
                    # 本节点若自行触发了摘要（plan_summary 缺失时的兜底），同样推进游标，
                    # 避免 supervisor 下一轮把这一步的产出再合并一次。
                    # 关键：只有当本节点**确实新生成**了摘要时才推进游标；否则原样保留 supervisor
                    # 写回的值。若这里无条件覆盖成 None/0，下一轮 seen 会退化为 0 → 增量摘要失效，
                    # 又变回每轮全量重述（等于白改）。
                    "summary_obs_seen": (_obs_total(state) if summary_fresh else state.get("summary_obs_seen")),
                }

            except GraphBubbleUp:
                # HITL 中断（interrupt）或父图冒泡信号：必须原样上抛，让外层图暂停 + checkpoint，
                # 交由 Studio / Command(resume=...) 恢复；绝不能被下面的 except Exception 当成业务
                # 失败吞掉或标 failed（否则人工确认永远等不到、该步还会被误判失败）。
                raise

            except GraphRecursionError:
                logger.warning("Recursion detected, breaking loop")
                return {
                    "messages": [AIMessage(content="Task completed to avoid infinite loop.")],
                    "sender": agent.name,
                    # 递归超限同样算本步未成功：显式标 failed，避免被后续 _mark_progress 洗成 completed
                    "execution_plan": _mark_failed(_plan, _cur - 1),
                    "current_step": _cur,
                    "run_started_at": state.get("run_started_at"),
                }

            except Exception as e:
                # 解释器/线程池正在关闭（langgraph dev 热重载、Ctrl+C 杀进程）：重试必然继续失败
                # （2026-09-30 线上现象：15ms 内连打 3 次 Attempt failed，全是同一个 executor 拒绝错误），
                # 且会把一次外部击杀误记成业务失败。快速退出：不标 failed、不动 error_count，
                # 返回空更新保持 checkpoint 干净，该步维持 pending，待进程重启后续跑。
                if _is_interpreter_shutdown(e):
                    logger.warning(
                        f"[shutdown] interpreter/thread-pool shutting down; abort {agent.name} "
                        f"step {_cur} WITHOUT retries and WITHOUT marking failed: {e}"
                    )
                    return {}
                logger.error(f"Attempt {attempt + 1} failed for {agent.name}: {e}")
                if attempt == max_retries - 1:
                    # 关键：把本步标 failed 并写回 state。supervisor 在派发时已把 current_step +1，
                    # 若不显式标 failed，下一轮 _mark_progress 会按索引把它算成 completed，
                    # 于是"重试三次全败"的步骤在 UI/日志里反而显示成已完成，掩盖真实故障。
                    failed_plan = _mark_failed(_plan, _cur - 1)
                    # 最终失败：回滚到上一个快照
                    if state.get("snapshot_id"):
                        rollback_msg = _run_tool(restore_snapshot, state["snapshot_id"])
                        return {
                            "messages": [AIMessage(content=f"Error recovered via rollback: {rollback_msg}")],
                            "sender": "Recovery",
                            "error_count": state.get("error_count", 0) + 1,
                            "execution_plan": failed_plan,
                            "current_step": _cur,
                            "run_started_at": state.get("run_started_at"),
                        }
                    else:
                        return {
                            "messages": [AIMessage(content=f"Critical error after {max_retries} attempts: {e}. Please clarify your request.")],
                            "sender": "ErrorHandler",
                            "error_count": state.get("error_count", 0) + 1,
                            "execution_plan": failed_plan,
                            "current_step": _cur,
                            "run_started_at": state.get("run_started_at"),
                        }

                # 重试：清理部分状态，推进异常计数（while 循环手动自增）
                attempt += 1
                state["messages"] = state["messages"][-10:]  # 保留最近10条消息
                continue

    return node


def create_nodes() -> dict:
    """基于 create_agents() 批量产出节点字典 {节点名: 节点闭包}。"""
    agents = create_agents()
    return {name: create_resilient_node(agent) for name, agent in agents.items()}
