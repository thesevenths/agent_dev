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
import json
import logging
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

from llm import (
    ToolCallLoggingMiddleware, CustomContextMiddleware,
    chat_llm, db_llm, coder_llm, crawler_llm, rag_llm, context_engineer_llm,
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
    list_files_metadata,
    save_context_snapshot, list_context_snapshots, evaluate_output, restore_snapshot,
    _run_tool,
)
from context import _date_context_str, _data_freshness_check
from compress import _compress_messages, _msg_text
from handoff import _step_assignment_text, _save_artifact, TMP_DIR
from summary import _summarize_observations, _AGENT_SUMMARY_DISABLE, _obs_total
from plan import _mark_failed
from runlog import set_current, ensure_run, log_event

logger = logging.getLogger(__name__)


# === 创建 Agent（使用 LangChain 1.4.x create_agent + Middleware for Context Engineer）===
def create_resilient_agent(llm, tools, system_prompt, agent_name="Agent", middleware=None):
    """创建标准化 Agent with resilience"""
    system_msg = SystemMessagePromptTemplate.from_template(system_prompt)
    prompt = ChatPromptTemplate.from_messages([system_msg, MessagesPlaceholder(variable_name="messages")])
    # 工具调用日志中间件置于最外层（first defined = outermost），保证它包住其余中间件里的工具调用
    mw = [ToolCallLoggingMiddleware()] + list(middleware or [])
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
        tools=[read_file, create_file, str_replace, send_qq_email, get_current_time],
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
        tools=[python_repl, create_file, read_file, str_replace, shell_exec, get_current_time],
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
        tools=[list_files_metadata, read_file, get_current_time],
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
    def node(state: dict) -> dict:
        # 所有节点都能看到当前计划，增强可观测性（execution_plan 可能为 None，统一当成空列表）
        _plan = state.get("execution_plan") or []
        _cur = state.get("current_step", 0)
        _step = _plan[_cur - 1] if (_plan and 0 < _cur <= len(_plan)) else None
        if isinstance(_step, dict):
            _step_txt = f"{_step.get('title', '')} | {_step.get('description', '')}"
        else:
            _step_txt = str(_step) if _step is not None else "No plan"
        # 让本步内子 agent 触发的工具（如 tavily 检索）把日志落到正确的运行日志文件。
        _mk = state.get("memory_key")
        set_current(_mk)
        ensure_run(_mk)
        logger.info(f"Executing plan step {_cur}/{len(_plan)}: {_step_txt}")
        max_retries = 3
        for attempt in range(max_retries):
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
                result = agent.invoke(run_state)

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

                return {
                    "messages": result["messages"],
                    "sender": agent.name,
                    "error_count": 0,
                    "snapshot_id": state.get("snapshot_id"),
                    # 把计划状态原样带回，让 supervisor 能继续追踪进度（None 归一为空列表）
                    "execution_plan": state.get("execution_plan") or [],
                    "plan_goal": state.get("plan_goal"),
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

            except GraphRecursionError:
                logger.warning("Recursion detected, breaking loop")
                return {
                    "messages": [AIMessage(content="Task completed to avoid infinite loop.")],
                    "sender": agent.name,
                    # 递归超限同样算本步未成功：显式标 failed，避免被后续 _mark_progress 洗成 completed
                    "execution_plan": _mark_failed(_plan, _cur - 1),
                    "current_step": _cur,
                }

            except Exception as e:
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
                        }
                    else:
                        return {
                            "messages": [AIMessage(content=f"Critical error after {max_retries} attempts: {e}. Please clarify your request.")],
                            "sender": "ErrorHandler",
                            "error_count": state.get("error_count", 0) + 1,
                            "execution_plan": failed_plan,
                            "current_step": _cur,
                        }

                # 重试：清理部分状态
                state["messages"] = state["messages"][-10:]  # 保留最近10条消息
                continue

    return node


def create_nodes() -> dict:
    """基于 create_agents() 批量产出节点字典 {节点名: 节点闭包}。"""
    agents = create_agents()
    return {name: create_resilient_node(agent) for name, agent in agents.items()}
