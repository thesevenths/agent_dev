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

# === 统一产物输出目录：所有数据/报告文件写到 <project>/tmp，避免污染仓库根目录 ===
PROJECT_ROOT = Path(__file__).resolve().parent
TMP_DIR = PROJECT_ROOT / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)
# 供 tools.py 的 create_file/read_file/str_replace 解析相对路径（延迟读取，晚于本模块导入也没关系）
os.environ.setdefault("AGENT_TMP_DIR", str(TMP_DIR))

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

# === AgentState（增强版：支持快照、错误状态、结构化计划与reason）===
# 结构化步骤： Plan{goal, steps:[{status}]} 设计，
# 每步带 title/description/status，使"只重排未完成步"与"天然终止"有显式状态支撑。
class PlanStep(TypedDict):
    title: str
    description: str
    status: str  # "pending" | "completed"

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str | None
    next: str | None
    reason: str | None  # Added for supervisor reason
    error_count: int  # 错误计数，用于重试
    snapshot_id: str | None  # 当前快照 ID
    memory_key: str  # 对话线程 ID
    hallucination_check: bool | None  # 幻觉检查标志
    # 结构化执行计划（每步带 status；对应 demo 的 Plan.steps）
    execution_plan: Optional[List[PlanStep]]
    plan_goal: Optional[str]  # 计划目标；再规划时永不改变（对应 demo 的 plan.goal）
    observations: List  # 每步 ToolMessage + 总结 AIMessage 累积，作为再规划上下文（对应 demo 的 state["observations"]）
    current_step: int
    artifacts: List[str]  # 每步落盘的产物文件路径（tmp/ 下），作为 agent 间 handoff 的可靠通道

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
        # timeout 让后端挂掉/拒连时快速失败而非长时间挂起；过载时也能更快暴露而非堆积连接
        sync_http = httpx.Client(verify=False, timeout=60)
        async_http = httpx.AsyncClient(verify=False, timeout=60)
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            http_client=sync_http,
            http_async_client=async_http,
            max_retries=2,
        )
    return ChatOpenAI(
        model=model,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=temperature,
        timeout=60,
        max_retries=2,
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

# === 成员配置 ===
members = [
    "chat_agent", "code_agent", "db_agent", 
    "crawler_agent", "rag_agent", "context_engineer_agent"
]
options = members + ["FINISH"]

class Router(TypedDict):
    next: str
    reason: str  # Added for reason
    execution_plan: Optional[List[PlanStep]]   # 首次规划或再规划时输出（结构化步骤列表）
    goal: Optional[str]  # 仅首次规划时输出目标

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


def _normalize_step(s) -> dict:
    """把任意 step（dict 或旧版字符串）归一化为 {title, description, status} 结构化步骤。"""
    if isinstance(s, dict):
        status = s.get("status", "pending")
        if status not in ("pending", "completed"):
            status = "pending"
        return {
            "title": str(s.get("title", "")),
            "description": str(s.get("description", "")),
            "status": status,
        }
    # 兼容旧字符串格式：整段作为 description
    return {"title": "", "description": str(s), "status": "pending"}


def _normalize_plan(plan) -> list:
    """归一化整个 plan 为结构化步骤列表。"""
    if not isinstance(plan, list):
        return []
    return [_normalize_step(x) for x in plan]


def _parse_target_agent(step) -> str:
    """从计划步骤（dict 或字符串）解析目标 agent；解析失败回落 context_engineer_agent。"""
    text = ""
    if isinstance(step, dict):
        text = (str(step.get("title", "")) + " " + str(step.get("description", ""))).lower()
    elif isinstance(step, str):
        text = step.lower()
    for member in members:
        if member.replace("_agent", "") in text:
            return member
    return "context_engineer_agent"


def _goal_text(state: dict) -> str:
    """取首个用户消息作为原始目标。"""
    for m in state.get("messages", []) or []:
        if isinstance(m, HumanMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


def _latest_result_text(state: dict) -> str:
    """取最近若干条消息（AI 最终答复 / 工具结果）作为"上一步结果"上下文（observations 为空时的回落）。"""
    msgs = state.get("messages", []) or []
    parts = []
    for m in msgs[-8:]:
        if isinstance(m, AIMessage):
            c = m.content if isinstance(m.content, str) else str(m.content)
            parts.append("[agent final] " + c)
        elif isinstance(m, ToolMessage):
            c = m.content if isinstance(m.content, str) else str(m.content)
            parts.append("[tool] " + c[:600])
    return "\n".join(parts) if parts else "(no result yet)"


def _build_replan_context(state: dict) -> str:
    """再规划上下文：优先用 observations（每步 ToolMessage + 总结 AIMessage），无则回落最近 messages。

    对应 single-agent demo 的 update_planner_node 喂 state["observations"] 的做法——比只看
    "上一步单条文本"更完整，supervisor 能真正看到 crawler 只回了文本、数据落在文件里，
    从而正确改写后续 code 步。
    """
    obs = state.get("observations", []) or []
    if obs:
        parts = []
        for m in obs[-12:]:
            if isinstance(m, AIMessage):
                c = m.content if isinstance(m.content, str) else str(m.content)
                parts.append("[agent summary] " + c[:800])
            elif isinstance(m, ToolMessage):
                c = m.content if isinstance(m.content, str) else str(m.content)
                parts.append("[tool] " + c[:600])
        if parts:
            return "\n".join(parts)
    return _latest_result_text(state)


def _mark_progress(plan: list, current: int) -> list:
    """把已完成步（index < current）标记为 completed；其余保持 pending。

    独立于 LLM：即使再规划失败（后端不可达/解析异常），plan 的 status 也必须反映真实进度，
    否则 langgraph dev 里看到的永远是初始快照，"plan 没更新"无法归因。
    """
    new_plan = _normalize_plan(plan)
    for i in range(min(current, len(new_plan))):
        new_plan[i]["status"] = "completed"
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


def _replan_tail(state: dict, plan: list, current: int):
    """调用 LLM 审视并改写剩余步骤。返回 (current_agent, new_plan)。

    借鉴 single-agent demo 的 update_planner：
    - plan 为结构化步骤（含 status），只重写"未完成"的尾部（index > current），
      已完成步（index < current）保留并标记为 completed；
    - plan_goal 永不改变（对应 demo "don't change the goal"）；
    - 再规划上下文用 observations（每步 ToolMessage + 总结），比单条结果文本更完整。

    安全约束（防死循环）：
    - 已完成步 plan[:current] 永不改写，只标 completed；
    - 若模型返回的 plan 更长，截断到原长（只减不增），杜绝无限追加；
    - 模型未返回 execution_plan 时**保留原尾部**（曾错误地把尾部丢掉导致提前 FINISH）；
    - current_step 由调用方负责 +1，本函数不回退。
    任何解析/调用异常都向上抛，由调用方回落"按原计划下一步"。
    """
    goal = state.get("plan_goal") or _goal_text(state)
    latest = _build_replan_context(state)
    sys_prompt = supervisor_system_prompt.replace("{members}", ", ".join(members)) + "\n\n" + _date_context_str()
    sys_msg = SystemMessage(content=sys_prompt)
    before_view = _plan_view(_mark_progress(plan, current), current)
    logger.info(f"supervisor re-planning step {current + 1}/{len(plan)}; plan BEFORE:\n{before_view}")
    user_msg = HumanMessage(content=(
        f"User goal (NEVER change this):\n{goal}\n\n"
        f"Current execution_plan ('>>' marks the step being dispatched NOW, index {current}):\n{before_view}\n\n"
        f"Observations from completed steps (ToolMessages + summaries):\n{latest}\n\n"
        "Decide the agent for the CURRENT step, then revise ONLY the REMAINING steps (index > current) "
        "based on what actually happened. You may skip/merge/rewrite remaining steps, but you MUST: "
        "1) keep the goal unchanged; 2) NOT increase total plan length; 3) NOT re-run completed steps; "
        "4) keep each remaining step's 'status' as 'pending' (completed steps are already marked). "
        "Return strict JSON with 'next' (agent for current step or FINISH), 'reason', and "
        "'execution_plan' (full revised plan as objects with title/description/status)."
    ))
    messages = [sys_msg, user_msg]
    parsed = None
    try:
        resp = supervisor_llm.with_structured_output(Router).invoke(messages)
        parsed = dict(resp) if resp is not None else None
    except Exception as se:
        logger.warning(f"supervisor re-plan structured output failed ({se}); fallback to manual JSON parse")
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
    if next_agent == "FINISH":  # 还有步未执行却让 FINISH，回落当前步
        next_agent = _parse_target_agent(plan[current])

    # 基线：已完成步标 completed，未完成步沿用原计划（模型不给新计划时这就是最终结果）。
    new_plan = _mark_progress(plan, current)
    revised = parsed.get("execution_plan")
    if isinstance(revised, list) and revised:
        tail = _normalize_plan(revised[current:])  # 模型掌控 current 及之后
        if tail:
            new_plan = new_plan[:current] + tail
            if len(new_plan) > len(plan):          # 只减不增：截断到原长
                new_plan = new_plan[:len(plan)]
            if len(new_plan) < current:            # 安全兜底
                new_plan = _mark_progress(plan, current)
    logger.info(f"supervisor re-plan result: agent={next_agent}; plan AFTER:\n{_plan_view(new_plan)}")
    return next_agent, new_plan


def supervisor(state: AgentState) -> Dict[str, Any]:
    """Supervisor：支持一次性规划 + 多轮顺序执行"""
    try:
        # 情况1：已有执行计划 → 自适应再规划（每步根据上一步结果审视/改写剩余步骤）
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

            step_text = plan[current]
            target_agent = _parse_target_agent(step_text)
            try:
                # 调用 LLM 审视剩余步骤（可跳过/改写已失效步骤），带防死循环硬约束
                target_agent, plan = _replan_tail(state, plan, current)
            except Exception as re:
                logger.warning(f"supervisor re-plan failed ({re}); follow original plan step {current + 1}")
                # 回落：保持原计划仅推进当前步，但 status 必须照样反映进度（不依赖 LLM）
                target_agent, plan = target_agent, _mark_progress(plan, current)
            return {
                "next": target_agent,
                "reason": f"Following execution plan step {current + 1}/{len(plan)}: {step_text}",
                "execution_plan": plan,
                "plan_goal": state.get("plan_goal"),  # 原样带回，再规划不改 goal
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
                logger.info(f"Supervisor created execution plan (goal={goal!r}):\n" + "\n".join(
                    f"{i+1}. {s.get('title','')}: {s.get('description','')}" for i, s in enumerate(plan)
                ))
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

# === 子 Agent 任务分发 & 上下文压缩 ===
# 背景（本轮两个线上问题的根因）：
# 1) supervisor 之前只把 structure plan 存在 state 里，节点喂给子 agent 的仍是 state["messages"]
#    ——里面只有用户的原始 query。子 agent 完全不知道"这一步到底要它干什么"，于是各自对着原始
#    query 从头重做：crawler 搜完，code/chat 又用同样的 query 再搜一遍。
# 2) 上游结果全靠 messages 传递，随步数线性膨胀，且工具原始输出（搜索快照/HTML）体积最大。
# 对策：
#   a) 每步把 supervisor 的当前步指令显式注入到子 agent（第 N/M 步 + title/description +
#      上游产物路径），并明确禁止重跑上游已完成的工作；
#   b) 每步把最终输出落盘到 tmp/，用"文件路径"作为跨 agent handoff 的可靠通道——文件不受
#      context 窗口限制，下游 read_file 即可拿到全量数据；
#   c) 送入子 agent 前对历史消息做 pair-safe 压缩（工具输出截断 + 旧消息按窗口裁剪）。
_CTX_TOOL_CHARS = int(os.environ.get("AGENT_CTX_TOOL_CHARS", 700))       # 单条 ToolMessage 保留上限
_CTX_AI_CHARS = int(os.environ.get("AGENT_CTX_AI_CHARS", 1200))          # 历史 AI 消息保留上限
_CTX_RECENT_AI_CHARS = int(os.environ.get("AGENT_CTX_RECENT_AI_CHARS", 6000))  # 最近一条上游结果上限
_CTX_KEEP_LAST = int(os.environ.get("AGENT_CTX_KEEP_LAST_MSGS", 24))     # 送入子 agent 的消息条数上限


def _msg_text(m) -> str:
    c = getattr(m, "content", "")
    return c if isinstance(c, str) else str(c)


def _truncate(text: str, limit: int) -> str:
    if limit > 0 and len(text) > limit:
        return text[:limit] + f"\n...[truncated {len(text) - limit} chars; full content kept in state.observations / tmp artifact]"
    return text


def _compress_messages(msgs, keep_last: int = _CTX_KEEP_LAST):
    """pair-safe 压缩消息历史：保 首条用户 query + 最近 keep_last 条，其余按完整工具组裁剪。

    - 工具输出（体积最大）一律截断到 _CTX_TOOL_CHARS；
    - 历史 AI 消息截断到 _CTX_AI_CHARS 并加来源标签；最近一条（上一步的正式产出）保留更大预算；
    - 裁剪窗口时绝不切断 AIMessage(tool_calls) → ToolMessage 配对（否则部分推理服务会报
      "tool_call_id 无对应 assistant 消息"），遇到 ToolMessage 边界就后移到安全位置。
    """
    msgs = list(msgs or [])
    if not msgs:
        return msgs
    last_ai_idx = -1
    for i, m in enumerate(msgs):
        if isinstance(m, AIMessage):
            last_ai_idx = i
    out = []
    for i, m in enumerate(msgs):
        if isinstance(m, ToolMessage):
            out.append(ToolMessage(
                content=_truncate(_msg_text(m), _CTX_TOOL_CHARS),
                tool_call_id=getattr(m, "tool_call_id", None),
                name=getattr(m, "name", None),
            ))
        elif isinstance(m, AIMessage):
            limit = _CTX_RECENT_AI_CHARS if i == last_ai_idx else _CTX_AI_CHARS
            who = getattr(m, "name", None) or "upstream agent"
            label = "[Most recent upstream result] " if i == last_ai_idx else f"[Earlier output from {who}] "
            if not getattr(m, "tool_calls", None):  # 纯文本答复才打标签，避免破坏 tool_call 消息结构
                out.append(AIMessage(content=label + _truncate(_msg_text(m), limit), name=m.name))
            else:
                out.append(m)
        else:
            out.append(m)

    # 窗口裁剪（pair-safe）
    if len(out) > keep_last:
        cut = len(out) - keep_last
        while cut < len(out) and isinstance(out[cut], ToolMessage):
            cut += 1  # 不要落在工具组的中间
        head = [out[0]] if (isinstance(out[0], HumanMessage) and cut > 0) else []
        out = head + out[cut:]
    return out


def _step_assignment_text(state: dict, agent_name: str) -> str:
    """构造"本步任务指令"：告诉子 agent 它在计划的第几步、要干什么、上游产物在哪、不许重做上游工作。"""
    plan = _normalize_plan(state.get("execution_plan") or [])
    cur = state.get("current_step", 0)
    step = plan[cur - 1] if (plan and 0 < cur <= len(plan)) else None
    lines = [_date_context_str(), ""]
    if step:
        lines += [
            f"[Supervisor assignment] You are executing step {cur}/{len(plan)} of an approved multi-agent plan.",
            f"Overall goal: {state.get('plan_goal') or _goal_text(state)}",
            f"This step is assigned to: {step.get('description', '')}",
            f"Step title: {step.get('title', '')}",
        ]
    else:
        lines.append("[Supervisor assignment] No plan step recorded; answer the user's latest request directly.")
    artifacts = state.get("artifacts") or []
    if artifacts:
        lines.append("")
        lines.append("Upstream results are ALREADY available as persisted files (read them with read_file if you need the full data):")
        for p in artifacts[-5:]:
            lines.append(f"  - {p}")
    lines += [
        "",
        "HARD RULES for this step:",
        "1. Do NOT repeat work already done by upstream agents (e.g. do NOT re-run the same web search / re-crawl). "
        "Their outputs are in this conversation and in the files listed above—consume them as your INPUT.",
        "2. Output ONLY the deliverable for THIS step, then stop. Do not answer parts of the goal belonging to other steps.",
        "3. Save substantial results/reports/data to a file and state the file path in your reply.",
        f"4. All files you write must go to this directory: {TMP_DIR}",
    ]
    return "\n".join(lines)


def _save_artifact(agent_name: str, step_idx: int, content: str) -> str | None:
    """把本步的最终输出落盘到 tmp/，作为下游 handoff 的可靠通道（不受 context 窗口限制）。"""
    try:
        if not content or not str(content).strip():
            return None
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        safe_agent = re.sub(r"[^A-Za-z0-9_.-]", "", agent_name or "agent")
        path = TMP_DIR / f"{ts}__step{step_idx}__{safe_agent}.md"
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(content))
        logger.info(f"artifact saved: {path}")
        return str(path)
    except Exception as e:
        logger.warning(f"save artifact failed: {e}")
        return None


# === 通用 Agent 节点（带错误恢复，集成 Middleware行为）===
def create_resilient_node(agent):
    """创建带错误恢复的节点函数"""
    def node(state: AgentState) -> Dict[str, Any]:
        # 所有节点都能看到当前计划，增强可观测性（execution_plan 可能为 None，统一当成空列表）
        _plan = state.get("execution_plan") or []
        _cur = state.get("current_step", 0)
        _step = _plan[_cur - 1] if (_plan and 0 < _cur <= len(_plan)) else None
        if isinstance(_step, dict):
            _step_txt = f"{_step.get('title', '')} | {_step.get('description', '')}"
        else:
            _step_txt = str(_step) if _step is not None else "No plan"
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
                assignment = _step_assignment_text(state, agent.name)
                run_state["messages"] = compressed + [HumanMessage(content=assignment)]
                logger.info(f"{agent.name} step assignment:\n{assignment[:400]}")
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

                # 产物落盘：把本步最终答复写到 tmp/ 并记入 artifacts，作为下游 handoff 通道。
                # 大块数据走文件而非 context，既避免 context 膨胀，也保证下游能拿到全量而非截断版。
                final_text = ""
                for m in reversed(new_obs):
                    if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
                        final_text = _msg_text(m)
                        break
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
                    "artifacts": artifacts,
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