"""
Multi-Agent System with Memory, Rollback, and Visualization
- 6 Agents: Chat, Code, DB, Crawler, RAG, Context Engineer
- Memory: SQLite Checkpointer for conversation history
- Snapshots: Visualized as PNG/HTML with Mermaid diagrams
- Error Recovery: Automatic retry + fallback to other agents
- Adapted to LangChain 1.4.x / LangGraph 1.x middleware API (create_agent + AgentMiddleware + new wrap_* signatures)
"""

import sys
import os
import json
import re
from langgraph.checkpoint.memory import MemorySaver
from datetime import datetime, timedelta
from typing import Annotated, Sequence, Dict, Any, Optional, Callable, List
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langgraph.errors import GraphRecursionError
import operator
import logging
from pathlib import Path
from typing import Annotated, Sequence, Dict, Any, Optional, Callable, List, List

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# === 配置 ===
from dotenv import load_dotenv
load_dotenv()

# 历史在线模型配置（已迁移到 .env 的 LLM_* 变量，保留备用，勿删除）：
#   DASHSCOPE_API_KEY  /  DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
#   切回在线模型时把 .env 里的 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL 改成对应值即可
from config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, LLM_VERIFY_SSL

# 调试
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === 导入 Prompt 和 Tools ===
from prompt import (
    db_system_prompt, supervisor_system_prompt, rag_system_prompt, 
    agentic_context_system_prompt, crawler_system_prompt, coder_system_prompt, chat_system_prompt
)
from tools import (
    # Chat tools
    read_file, create_file, str_replace, send_qq_email,
    # DB tools
    add_sale, delete_sale, update_sale, query_sales, query_table_schema, execute_sql,
    # Code tools
    python_repl, shell_exec,
    # Crawler tools
    get_nasdaq_top_gainers, get_crypto_sentiment_indicators, resilient_tavily_search,
    # 时间工具（所有 agent 都可能需锚定相对时间到"今天"，统一提供）
    get_current_time,
    # RAG tools
    list_files_metadata,
    # Context tools
    save_context_snapshot, list_context_snapshots, evaluate_output, restore_snapshot
)

# LangChain 1.4.x Imports for Agents and Middleware
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware, SummarizationMiddleware, HumanInTheLoopMiddleware,
    ModelRequest, ModelResponse, ToolCallRequest,
)
from langchain_openai import ChatOpenAI

# === 工具调用辅助 ===
# @tool 装饰的 StructuredTool 在 langchain-core>=1.6 已不可直接调用（tool(...) 会抛
# TypeError: 'StructuredTool' object is not callable）。这里统一走底层 .func 调用，
# 供 middleware / 节点 / invoke_with_memory 调用这些工具时使用，避免把工具当普通函数直接调。
def _run_tool(tool, *args, **kwargs):
    func = getattr(tool, "func", None)
    if callable(func):
        return func(*args, **kwargs)
    if hasattr(tool, "invoke") and kwargs:
        return tool.invoke(kwargs)
    return tool(*args, **kwargs)

# === 当前日期上下文（根治：模型无实时时钟，必须显式注入"今天/昨天"）===
# 否则模型会把"昨天"映射到训练记忆里的某次事件（如把 A 股大跌猜成 2025年6月）。
_WEEKDAY_CN = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

def _date_context_str() -> str:
    """返回 [System context] 串，含今天/昨天的精确日期，注入给各 agent 以解析'昨天/上周/本月'等相对时间。"""
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    return (
        f"[System context] Today's date is {today.strftime('%Y-%m-%d')} ({_WEEKDAY_CN[today.weekday()]}). "
        f"Yesterday was {yesterday.strftime('%Y-%m-%d')} ({_WEEKDAY_CN[yesterday.weekday()]}). "
        f"Use these EXACT dates to resolve any relative time expression (昨天/上周/本月/近期) in the user request. "
        f"Do NOT guess or invent the year/month/day."
    )

# === AgentState（增强版：支持快照、错误状态和reason）===
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str | None
    next: str | None
    reason: str | None  # Added for supervisor reason
    error_count: int  # 错误计数，用于重试
    snapshot_id: str | None  # 当前快照 ID
    memory_key: str  # 对话线程 ID
    hallucination_check: bool | None  # 幻觉检查标志
    execution_plan: Optional[List[str]]     
    current_step: int

# === LLMs 配置 ===
def create_llm(temperature=0.1, model_name=None):
    """创建统一的 LLM（兼容本地 vLLM / 在线 DashScope 等 OpenAI 兼容接口）

    连接信息全部来自 .env 的 LLM_* 变量，改配置文件即可换模型，无需改代码。
    """
    model = model_name or LLM_MODEL
    # 本地服务若为自签名/内网证书，LLM_VERIFY_SSL=false 等价于 curl -k。
    # 注意：langchain-openai 会同时构建同步与异步 httpx 客户端，二者都需显式给
    # api_key/base_url 凭证，否则异步客户端会从 OPENAI_API_KEY 环境变量读取而报
    # "Missing credentials"。这里在两条分支都传入 api_key/base_url，并在关闭 SSL
    # 校验时同时提供 http_client / http_async_client（均 verify=False）。
    if not LLM_VERIFY_SSL:
        import httpx
        sync_http = httpx.Client(verify=False)
        async_http = httpx.AsyncClient(verify=False)
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            http_client=sync_http,
            http_async_client=async_http,
        )
    return ChatOpenAI(
        model=model,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=temperature,
    )

supervisor_llm = create_llm(temperature=0.0)
chat_llm = create_llm()
db_llm = create_llm(temperature=0.0)  # DB 需要确定性
coder_llm = create_llm(temperature=0.3)  # 代码生成需要创造性
crawler_llm = create_llm()
rag_llm = create_llm(temperature=0.1)
context_engineer_llm = create_llm(temperature=0.2)

# === 自定义 Middleware for Context Engineer ===
class CustomContextMiddleware(AgentMiddleware):
    """Context Engineer 自定义中间件（适配 LangChain 1.4.x middleware API）。

    1.4.x 变更要点（相对 1.0.x）：
    - before_model / after_model / after_agent 接收 (state, runtime) 并返回 state 更新 dict 或 None，
      不再接收/返回 request/response 对象。
    - wrap_tool_call 返回 ToolMessage | Command（旧版 ToolCallResponse 已移除）。
    - Runtime 为 frozen dataclass，无法挂载自定义字段；快照 ID 改存为实例属性 self._last_snapshot_id。
    - 消息历史通过 request.state["messages"] 读取（Runtime 不再持有 messages）。
    """

    def __init__(self):
        super().__init__()
        self._last_snapshot_id: str | None = None

    def before_model(self, state, runtime):
        # 动态上下文注入：1.4.x 下 messages 由 add_messages reducer 合并，无法在此直接裁剪历史，
        # 这里仅做只读相关性检查，保持无副作用，避免破坏消息合并语义。
        logger.info("CustomContextMiddleware.before_model: 动态上下文注入（1.4.x 下为只读检查）")
        return None

    def after_model(self, state, runtime):
        # 上下文评估 & 压缩：评估模型输出，失败时告警；压缩交由流水线中的 SummarizationMiddleware 处理。
        messages = state.get("messages", []) if isinstance(state, dict) else getattr(state, "messages", [])
        last_msg = messages[-1] if messages else None
        content = ""
        if last_msg is not None:
            raw = getattr(last_msg, "content", "")
            content = " ".join(str(c) for c in raw) if isinstance(raw, list) else (raw or "")
        try:
            eval_result = _run_tool(evaluate_output, "Correctness;Completeness;No Hallucination", content)
        except Exception as ee:
            logger.warning(f"Output evaluation skipped (tool error): {ee}")
            eval_result = {"passed": True, "reason": "eval skipped"}
        if not eval_result.get("passed", False):
            logger.warning(f"Output evaluation failed: {eval_result.get('reason')}")
        logger.info("Context evaluated and compressed if needed.")
        return None

    def wrap_tool_call(self, request, handler):
        # 工具调用前保存快照，出错时回滚。
        tool_call = request.tool_call or {}
        tc_name = tool_call.get("name") if isinstance(tool_call, dict) else None
        tc_id = tool_call.get("id") if isinstance(tool_call, dict) else None
        st = getattr(request, "state", None)
        msgs = []
        if isinstance(st, dict):
            msgs = st.get("messages", []) or []
        elif st is not None:
            msgs = getattr(st, "messages", []) or []
        payload = {
            "messages": [getattr(m, "content", "") for m in msgs[-5:]],
            "tool": tc_name,
            "timestamp": datetime.now().isoformat(),
        }
        snap = _run_tool(save_context_snapshot, name=f"tool_{tc_name}", content=json.dumps(payload, ensure_ascii=False))
        snapshot_id = snap.get("path") if isinstance(snap, dict) else None
        if isinstance(snapshot_id, str):
            snapshot_id = os.path.splitext(os.path.basename(snapshot_id))[0]
        self._last_snapshot_id = snapshot_id
        logger.info(f"Pre-tool snapshot saved: {snapshot_id}")

        try:
            result = handler(request)
        except Exception as e:
            # Error Recovery: Rollback on error/hallucination
            logger.error(f"Tool call error: {e}. Rolling back.")
            if self._last_snapshot_id:
                try:
                    _run_tool(restore_snapshot, self._last_snapshot_id)
                except Exception as re:
                    logger.error(f"Rollback failed: {re}")
            result = ToolMessage(
                content=f"Tool call failed and rolled back: {e}",
                tool_call_id=tc_id,
                name=tc_name,
            )
        return result

    def wrap_model_call(self, request, handler):
        # 模型生成阶段出错时回滚到最近快照。
        try:
            return handler(request)
        except Exception as e:
            logger.error(f"Model call error: {e}. Attempting recovery.")
            if self._last_snapshot_id:
                try:
                    _run_tool(restore_snapshot, self._last_snapshot_id)
                except Exception as re:
                    logger.error(f"Rollback failed: {re}")
            return AIMessage(content=f"Recovered from error: {e}")

    def after_agent(self, state, runtime):
        # Agent 结束后可视化最近快照。
        if self._last_snapshot_id:
            viz = globals().get("visualize_snapshot")
            if viz is not None:
                try:
                    viz(self._last_snapshot_id)
                except Exception as e:
                    logger.warning(f"visualize_snapshot failed: {e}")
        return None

# === 创建 Agent（使用 LangChain 1.4.x create_agent + Middleware for Context Engineer）===
def create_resilient_agent(llm, tools, system_prompt, agent_name="Agent", middleware=None):
    """创建标准化 Agent with resilience"""
    system_msg = SystemMessagePromptTemplate.from_template(system_prompt)
    prompt = ChatPromptTemplate.from_messages([system_msg, MessagesPlaceholder(variable_name="messages")])
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,  # Passed directly in 1.4.x
        middleware=middleware or [],
        name=agent_name
    )

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
code_agent = create_resilient_agent(
    coder_llm,
    tools=[python_repl, create_file, read_file, str_replace, shell_exec, resilient_tavily_search, get_current_time],
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

# === 成员配置 ===
members = [
    "chat_agent", "code_agent", "db_agent", 
    "crawler_agent", "rag_agent", "context_engineer_agent"
]
options = members + ["FINISH"]

class Router(TypedDict):
    next: str
    reason: str  # Added for reason
    execution_plan: Optional[List[str]]   # 只有第一次规划时才输出

# === Supervisor（支持错误恢复 + Reason Output）===
def _extract_json_obj(text):
    """从模型文本输出中稳健抽取第一个 JSON 对象（兼容 ```json 围栏与前后多余文本）。"""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if "```" in text:
        m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    try:
        return json.loads(text)
    except Exception:
        return {}


def supervisor(state: AgentState) -> Dict[str, Any]:
    """Supervisor：支持一次性规划 + 多轮顺序执行"""
    try:
        # 情况1：已经有执行计划 → 严格按计划走（第2~N轮）
        plan = state.get("execution_plan") or []
        if plan and len(plan) > 0:
            current = state.get("current_step", 0)
            if current >= len(plan):
                # 所有步骤都完成了
                return {
                    "next": "FINISH",
                    "reason": "All tasks in execution plan completed.",
                    "current_step": current
                }
            
            # 解析当前步骤应该交给哪个 agent
            step_text = plan[current]
            # 简单解析 "数字. 任务 → agent" 格式
            target_agent = None
            for member in members:
                if member.replace("_agent", "") in step_text.lower():
                    target_agent = member
                    break
            if not target_agent:
                target_agent = "context_engineer_agent"  # 兜底

            return {
                "next": target_agent,
                "reason": f"Following execution plan step {current+1}/{len(plan)}: {step_text}",
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
            
            # 如果模型给出了计划，就采纳
            plan = response.get("execution_plan")
            if plan and len(plan) > 0:
                logger.info(f"Supervisor created execution plan:\n" + "\n".join(plan))
                # 第一步立刻执行
                first_agent = "context_engineer_agent"  # 兜底
                for member in members:
                    if member.replace("_agent", "") in plan[0].lower():
                        first_agent = member
                        break
                return {
                    "next": first_agent,
                    "reason": f"Starting execution plan step 1/{len(plan)}: {plan[0]}",
                    "execution_plan": plan,
                    "current_step": 1
                }
            else:
                # 降级为传统单轮路由（兼容旧逻辑）
                return {
                    "next": response["next"],
                    "reason": response["reason"] + " (no multi-step plan generated)"
                }

    except Exception as e:
        logger.error(f"Supervisor error: {e}")
        return {
            "next": "context_engineer_agent",
            "reason": f"Supervisor fallback due to error: {e}",
            "error_count": state.get("error_count", 0) + 1
        }

# === 通用 Agent 节点（带错误恢复，集成 Middleware行为）===
def create_resilient_node(agent):
    """创建带错误恢复的节点函数"""
    def node(state: AgentState) -> Dict[str, Any]:
        # 所有节点都能看到当前计划，增强可观测性（execution_plan 可能为 None，统一当成空列表）
        _plan = state.get("execution_plan") or []
        _cur = state.get("current_step", 0)
        _step_txt = _plan[_cur - 1] if (_plan and 0 < _cur <= len(_plan)) else "No plan"
        logger.info(f"Executing plan step {_cur}/{len(_plan)}: {_step_txt}")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 执行 Agent (LangChain 1.0 invoke)
                # 注入当前日期上下文：模型无实时时钟，必须显式告知"今天/昨天"才能正确解析
                # 相对时间（如"昨天"），否则会瞎猜年份。用副本注入，结果里剥离避免污染全局 state。
                date_ctx = SystemMessage(content=_date_context_str())
                run_state = dict(state)
                run_state["messages"] = [date_ctx] + list(state.get("messages", []))
                result = agent.invoke(run_state)
                # 剥离注入的日期 SystemMessage（agent 通常会原样回传输入系统消息），避免累积进全局 state
                if isinstance(result, dict) and "messages" in result:
                    result = {
                        **result,
                        "messages": [
                            m for m in result["messages"]
                            if not (isinstance(m, SystemMessage) and m.content == date_ctx.content)
                        ],
                    }
                
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
                
                return {
                    "messages": result["messages"],
                    "sender": agent.name,
                    "error_count": 0,
                    "snapshot_id": state.get("snapshot_id"),
                    # 把计划状态原样带回，让 supervisor 能继续追踪进度（None 归一为空列表）
                    "execution_plan": state.get("execution_plan") or [],
                    "current_step": state.get("current_step", 0),
                }
                
            except GraphRecursionError:
                logger.warning("Recursion detected, breaking loop")
                return {"messages": [AIMessage(content="Task completed to avoid infinite loop.")], "sender": agent.name}
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed for {agent.name}: {e}")
                if attempt == max_retries - 1:
                    # 最终失败：回滚到上一个快照
                    if state.get("snapshot_id"):
                        rollback_msg = _run_tool(restore_snapshot, state["snapshot_id"])
                        return {
                            "messages": [AIMessage(content=f"Error recovered via rollback: {rollback_msg}")],
                            "sender": "Recovery",
                            "error_count": state.get("error_count", 0) + 1
                        }
                    else:
                        return {
                            "messages": [AIMessage(content=f"Critical error after {max_retries} attempts: {e}. Please clarify your request.")],
                            "sender": "ErrorHandler",
                            "error_count": state.get("error_count", 0) + 1
                        }
                
                # 重试：清理部分状态
                state["messages"] = state["messages"][-10:]  # 保留最近10条消息
                continue
    
    return node

# === 创建节点 ===
chat_node = create_resilient_node(chat_agent)
db_node = create_resilient_node(db_agent)
code_node = create_resilient_node(code_agent)
crawler_node = create_resilient_node(crawler_agent)
rag_node = create_resilient_node(rag_agent)
context_node = create_resilient_node(context_engineer)  # Middleware applied here

# === 构建 Graph（带记忆）===
def build_graph_with_memory():
    """构建 Graph（Checkpointer 按运行模式自动决定）

    - 直接运行（python agent.py / invoke_with_memory）：挂 MemorySaver，保留跨轮对话记忆；
    - 经 LangGraph API（langgraph dev / langgraph up）加载时，平台自带持久化，
      graph 不能带自定义 checkpointer，否则 dev 服务报 ValueError 拒绝加载。
      用 LANGSMITH_LANGGRAPH_API_VARIANT 环境变量识别，编译时不挂自定义 checkpointer。
    """
    # 初始化 Checkpointer（SQLite 记忆）
    os.makedirs("./memory", exist_ok=True)
    memory = MemorySaver()
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("chat_agent", chat_node)
    workflow.add_node("db_agent", db_node)
    workflow.add_node("code_agent", code_node)
    workflow.add_node("crawler_agent", crawler_node)
    workflow.add_node("rag_agent", rag_node)
    workflow.add_node("context_engineer_agent", context_node)
    
    # 边：Agent → Supervisor
    for member in members:
        workflow.add_edge(member, "supervisor")
    
    # START → Supervisor
    workflow.add_edge(START, "supervisor")
    
    # 条件边
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {
            "chat_agent": "chat_agent",
            "db_agent": "db_agent",
            "code_agent": "code_agent",
            "crawler_agent": "crawler_agent",
            "rag_agent": "rag_agent",
            "context_engineer_agent": "context_engineer_agent",
            "FINISH": END,
        }
    )
    
    # 编译：
    # - 直接运行（python agent.py / invoke_with_memory）：挂 MemorySaver 以保留跨轮记忆；
    # - 经 LangGraph API（langgraph dev / langgraph up）加载时，平台自带持久化，
    #   graph 不能带自定义 checkpointer（否则 dev 服务报 ValueError 拒绝加载），故不挂。
    if os.environ.get("LANGSMITH_LANGGRAPH_API_VARIANT"):
        graph = workflow.compile()            # 无 checkpointer，由 LangGraph 平台接管 persistence
    else:
        graph = workflow.compile(checkpointer=memory)
    graph.name = "Resilient Multi-Agent System"
    return graph, memory

# === 快照可视化工具 ===
def visualize_snapshot(snapshot_id: str, output_dir: str = "./snapshots"):
    """可视化快照：生成 Mermaid PNG + HTML"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # 假设快照包含消息流
        snapshot_data = json.loads(open(f"./contexts/{snapshot_id}.json").read())
        messages = snapshot_data.get("messages", [])
        
        # 生成 Mermaid 流程图
        mermaid_code = "graph TD\n"
        for i, msg in enumerate(messages):
            sender = msg.get("sender", "Unknown")
            content = msg[:50] + "..." if len(msg) > 50 else msg  # 截断
            node_id = f"N{i}"
            mermaid_code += f'    {node_id}["{sender}: {content}"]\n'
            if i > 0:   
                mermaid_code += f"    N{i-1} --> {node_id}\n"
        
        # 保存 Mermaid
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head><script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script></head>
        <body>
            <div class="mermaid">
                {mermaid_code}
            </div>
            <script>mermaid.initialize({{startOnLoad:true}});</script>
        </body>
        </html>
        """
        
        html_path = f"{output_dir}/{snapshot_id}.html"
        png_path = f"{output_dir}/{snapshot_id}.png"  # 需要额外工具生成 PNG
        
        with open(html_path, "w") as f:
            f.write(html_content)
        
        logger.info(f"Snapshot visualized: {html_path}")
        return html_path
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        return None

# === 全局 Graph ===
graph, memory = build_graph_with_memory()

# === 工具函数：带记忆的调用 ===
def invoke_with_memory(query: str, thread_id: str = None, config: Optional[Dict] = None):
    """带记忆的 Graph 调用，支持回滚"""
    if thread_id is None:
        thread_id = str(datetime.now().timestamp())
    
    config = config or {"configurable": {"thread_id": thread_id}}
    
    try:
        # 流式执行（实时输出）
        final_state = None
        for chunk in graph.stream(
            {"messages": [HumanMessage(content=query)], "memory_key": thread_id},
            config=config
        ):
            print(chunk) 
            final_state = chunk
        
        # 可视化最终快照 (middleware already handles, but fallback)
        if final_state and final_state.get("snapshot_id"):
            viz_path = visualize_snapshot(final_state["snapshot_id"])
            if viz_path:
                print(f"📊 Snapshot visualization: {viz_path}")
        
        return final_state
        
    except Exception as e:
        logger.error(f"Invocation failed: {e}")
        # 紧急回滚：恢复到最新快照
        snap_result = _run_tool(list_context_snapshots)
        files = snap_result.get("snapshots", []) if isinstance(snap_result, dict) else []
        if files:
            latest = files[0]  # list_context_snapshots 已按时间倒序，files[0] 为最新
            rollback_msg = _run_tool(restore_snapshot, latest)
            print(f"🚨 Emergency rollback: {rollback_msg}")
        raise

# === 测试 ===
if __name__ == "__main__":
    # 初始化上下文目录
    os.makedirs("./contexts", exist_ok=True)
    os.makedirs("./snapshots", exist_ok=True)
    os.makedirs("./documents", exist_ok=True)
    
    # 测试 1：简单对话
    print("=== 测试 1：简单对话 ===")
    result1 = invoke_with_memory("你好，我是金融分析师")
    print(f"Final response: {result1['messages'][-1].content if result1 else 'Failed'}")
    
    # 测试 2：复杂查询（触发工具 + 错误恢复）
    print("\n=== 测试 2：纳斯达克查询 + 模拟错误 ===")
    try:
        # 模拟一个可能出错的查询
        result2 = invoke_with_memory("分析今天纳斯达克涨幅前3的股票，生成报告。如果出错请自动恢复。")
        print(f"Success: {result2['messages'][-1].content[:100] if result2 else 'Failed'}...")
    except Exception as e:
        print(f"Expected error handled: {e}")
    
    # 测试 3：加载记忆
    print("\n=== 测试 3：加载记忆继续对话 ===")
    thread_id = "test_thread_123"
    invoke_with_memory("之前我问了纳斯达克，现在帮我查数据库里的销售数据", thread_id=thread_id)
    
    print("\n🎉 Multi-Agent System with Memory & Recovery is ready!")
    print("Run: result = invoke_with_memory('your query', thread_id='unique_id')")