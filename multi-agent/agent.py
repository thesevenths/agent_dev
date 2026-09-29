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
import time
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

# 每轮运行的关键日志（supervisor 规划 / 再规划 / 子 agent 任务下发 / 联网检索）持久化到
# multi-agent/log/，文件名带时间戳，便于事后排查（不依赖 langgraph dev 终端）。
from runlog import start_run, ensure_run, set_current, log_event, LOG_DIR, write_summary

# === 当前日期上下文（根治：模型无实时时钟，必须显式注入"今天/昨天"）===
# 否则模型会把"昨天"映射到训练记忆里的某次事件（如把 A 股大跌猜成 2025年6月）。
_WEEKDAY_CN = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

def _date_context_str() -> str:
    """返回 [System context] 串，含今天/昨天的精确日期，注入给各 agent 以解析'昨天/上周/本月'等相对时间。

    除日期外还注入**当前时刻（HH:MM）与市场交易时段**：模型没有时钟，只给日期会让它
    在盘后（如 22:54）仍按"早盘/实时"去规划与检索，抓到 10:04 的盘中快照却当成当前行情
    做走势研判——数据滞后十几小时而全链路无任何提示。给时刻 + 时段，模型才能问对口径
    （收盘/收评 vs 盘中）。
    """
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    return (
        f"[System context] Today's date is {today.strftime('%Y-%m-%d')} ({_WEEKDAY_CN[today.weekday()]}). "
        f"Yesterday was {yesterday.strftime('%Y-%m-%d')} ({_WEEKDAY_CN[yesterday.weekday()]}). "
        f"Current local time is {today.strftime('%H:%M')}. "
        f"Use these EXACT dates to resolve any relative time expression (昨天/上周/本月/近期) in the user request. "
        f"Do NOT guess or invent the year/month/day.\n"
        f"{_market_session_note(today)}\n"
        f"[Data freshness rule] Any time-series data you retrieve or pass downstream MUST state its "
        f"as-of timestamp as a line 'AS_OF: YYYY-MM-DD HH:MM'. Before using such data, compare it with the "
        f"current local time above; if the data is from today but hours old, label it explicitly as "
        f"'as of HH:MM' and NEVER present it as current/real-time."
    )


# === 数据时效性 ===
# 交易时段表（名称 -> (开盘时,分), (收盘时,分)）。24h 市场（如 crypto）不列入，按"始终在市"处理。
_MARKET_SESSIONS = {"A股": ((9, 30), (15, 0))}
# "当日数据但滞后多少小时"才告警。历史数据（昨天及更早）不告警——用户要的就是历史口径，
# 若一并告警会把合法历史数据淹在误报里。
DATA_FRESHNESS_MAX_HOURS = float(os.environ.get("DATA_FRESHNESS_MAX_HOURS", "2"))


def _market_session_note(now: datetime) -> str:
    """给出当前时刻相对各市场的状态（盘前/盘中/已收盘）与取数口径建议。"""
    parts = []
    for name, ((oh, om), (ch, cm)) in _MARKET_SESSIONS.items():
        mins = now.hour * 60 + now.minute
        open_m, close_m = oh * 60 + om, ch * 60 + cm
        weekend = now.weekday() >= 5
        if weekend:
            state, tip = "休市（周末）", "取最近一个交易日的**收盘**数据"
        elif mins < open_m:
            state, tip = "盘前（未开盘）", "取上一交易日的**收盘**数据，不要取盘中数据"
        elif mins <= close_m:
            state, tip = "盘中", "可取实时/盘中数据，但必须标注 as-of 时刻（盘中数据会随时变化）"
        else:
            state, tip = ("已收盘", f"取当日**收盘/收评**数据；不要再把早盘/盘中快照当作当前行情"
                                    f"（距收盘已 {((mins - close_m) // 60)} 小时，会严重滞后）")
        parts.append(f"{name}：{state} → {tip}")
    return "[Market session] " + "；".join(parts)


_AS_OF_RE = re.compile(
    r"AS_OF\s*[:=]\s*(?:(?P<date>\d{4}-\d{2}-\d{2})\s+)?(?P<time>\d{1,2}:\d{2})", re.IGNORECASE
)


def _data_freshness_check(text: str, now: datetime | None = None) -> str | None:
    """检查产出文本中 AS_OF 标记的数据时刻，返回告警文案；无需告警则返回 None。

    设计取舍：**只校验"as_of 是今天但已滞后数小时"的情形**。
    用户明确要历史数据（如"昨天收盘"）时数据本就陈旧，告警属于误报，
    会把真正的时效问题淹没掉，故历史数据不告警。
    依赖模型按 prompt 规则输出 'AS_OF:' 标记；未输出则本函数静默返回 None（优雅降级）。
    """
    if not text:
        return None
    m = _AS_OF_RE.search(text)
    if not m:
        return None
    now = now or datetime.now()
    date_s = m.group("date") or now.strftime("%Y-%m-%d")
    try:
        as_of = datetime.strptime(f"{date_s} {m.group('time')}", "%Y-%m-%d %H:%M")
    except ValueError:
        return None
    if as_of.date() != now.date():
        return None  # 历史口径，合法，不告警
    delta_h = (now - as_of).total_seconds() / 3600.0
    if delta_h > DATA_FRESHNESS_MAX_HOURS:
        return (
            f"[数据时效] 产出标注的数据时刻 AS_OF {as_of.strftime('%Y-%m-%d %H:%M')} "
            f"距今已 {delta_h:.1f} 小时（阈值 {DATA_FRESHNESS_MAX_HOURS}h），"
            f"属当日滞后数据，不得作为当前/实时行情使用，下游引用时必须显式标注 as-of 时刻。"
        )
    if delta_h < -0.5:
        return f"[数据时效] 产出标注的数据时刻 {as_of.strftime('%Y-%m-%d %H:%M')} 晚于当前时间 {now.strftime('%H:%M')}，请核对时钟或数据源。"
    return None

# === AgentState（增强版：支持快照、错误状态、结构化计划与reason）===
# 结构化步骤： Plan{goal, steps:[{status}]} 设计，
# 每步带 title/description/status，使"只重排未完成步"与"天然终止"有显式状态支撑。
class PlanStep(TypedDict):
    title: str
    description: str
    status: str  # "pending" | "completed" | "failed"（failed = 重试耗尽/异常，终态，不会被索引推进洗成 completed）

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
    plan_summary: Optional[str]  # 已完成步的"要点清单"语义摘要（_summarize_observations 生成），作为下游 agent 的跨步上下文
    # 增量滚动摘要游标：observations 被截断到最近 40 条（从头部丢弃），故不能用 len(observations)
    # 判断是否"有新产出"，必须另存单调计数，否则截断后 len 不再增长 → 新步产出会被误判为"无新增"而跳过摘要。
    obs_total: int  # 单调递增：累计追加到 observations 的条数
    summary_obs_seen: int  # 已折叠进 plan_summary 的条数（= 上次摘要时的 obs_total）
    replan_noop_streak: int  # 连续"再规划空转"（BEFORE==AFTER）次数，达阈值后停用再规划

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

# === 工具调用日志（middleware 钩子，一处覆盖全部工具）===
def _tool_args_preview(args, limit: int = 300) -> str:
    try:
        s = json.dumps(args, ensure_ascii=False, default=str)
    except Exception:
        s = str(args)
    return s if len(s) <= limit else s[:limit] + f"...[+{len(s) - limit} chars]"


class ToolCallLoggingMiddleware(AgentMiddleware):
    """把每一次工具调用（名称 / 入参 / 耗时 / 结果体量 / 异常）写进运行日志与控制台。

    为什么需要：此前只有 tavily 在自己函数体内手打了日志（tools.py），而 python_repl /
    shell_exec / read_file / create_file / str_replace / send_qq_email 全无调用痕迹，
    于是 code_agent 连跑两三分钟在日志里完全黑盒——无法确认它到底执行了什么代码、
    声称生成的图表是否真的落盘，出问题只能靠猜。

    为什么用 middleware 的 wrap_tool_call 钩子，而不是逐个改 tools.py：
      1) 一处生效即覆盖全部工具（含以后新增的工具），不会出现"新工具忘了加日志"的漏网；
      2) 完全不触碰工具的 schema / 签名，零破坏风险。
    """

    def wrap_tool_call(self, request, handler):
        tc = getattr(request, "tool_call", None) or {}
        name = tc.get("name") or (request.tool.name if getattr(request, "tool", None) else "unknown")
        args = tc.get("args") or {}
        t0 = time.time()
        head = f"[tool] ▶ {name}({_tool_args_preview(args)})"
        logger.info(head)
        log_event(head)
        try:
            result = handler(request)
        except Exception as e:
            dur = time.time() - t0
            msg = f"[tool] ✖ {name} raised after {dur:.2f}s: {type(e).__name__}: {e}"
            logger.error(msg)
            log_event(msg)
            raise  # 不吞异常：重试/回滚逻辑依赖异常向上传播
        dur = time.time() - t0
        status = getattr(result, "status", "") or ""
        content = getattr(result, "content", "")
        content = content if isinstance(content, str) else str(content)
        flag = "✖" if status == "error" else "✔"
        tail = f"[tool] {flag} {name} ({dur:.2f}s, {len(content)} chars) -> {content[:200]}"
        logger.info(tail)
        log_event(tail)
        return result


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
        # status 合法取值：pending / completed / failed。failed 表示该步执行失败（重试耗尽/异常），
        # 是终态事实，不得被"索引推进"洗成 completed。
        status = s.get("status", "pending")
        if status not in ("pending", "completed", "failed"):
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


# === 跨步语义摘要（要点提取）===
# 背景：之前"下游 agent 拿不到上游结果"的根因是历史只靠 _compress_messages 粗暴截断 + observations
# 原样塞 40 条原始消息。截断会丢信息（用户担忧），原始 40 条又占 token 且不含"已检索到 X / 已存文件 Y /
# 结论 Z"这种提炼信息。故在 supervisor 每轮再规划时，把 observations 过一次 LLM 提炼成"要点清单"
# （key-points checklist），同时作为：①再规划上下文（比原始 observations 更准更省 token）；②下游 agent
# 的跨步语义上下文（与文件 handoff 共同承载信息，使原始历史可安全截断）；③落盘 log/<thread>_summary.md
# 供事后回看（对应 WorkBuddy/Qoder 把关键信息提取写入 md 的行为）。
_SUMMARY_PROMPT = (
    "You are condensing the COMPLETED steps of a multi-agent execution plan into a concise "
    "KEY-POINTS checklist for the downstream agents that will run next.\n\n"
    "For each completed step extract ONLY the decision-relevant essentials:\n"
    "- What the step accomplished (one line).\n"
    "- Key data / numbers / dates / facts it found — keep the ACTUAL values, not vague phrasing.\n"
    "- Files it persisted (absolute paths) — downstream agents MUST read these for full data; "
    "do NOT re-fetch or re-crawl.\n"
    "- Any conclusion or decision it reached.\n\n"
    "Rules:\n"
    "- Be concrete and faithful to the source; never invent data.\n"
    "- Total under ~400 words. Use a markdown bullet list.\n"
    "- Do NOT paste raw tool JSON or long excerpts; keep only what a later agent needs to act.\n"
)

_SUMMARY_MERGE_PROMPT = (
    "You are UPDATING an existing KEY-POINTS checklist that condenses the COMPLETED steps of a "
    "multi-agent execution plan. You are given:\n"
    "  1) the CURRENT checklist — it is ALREADY condensed and is the authoritative record of the "
    "     older steps, and\n"
    "  2) ONLY the NEW output of the most recently completed step.\n\n"
    "Produce the UPDATED checklist by MERGING — never by re-deriving the whole history:\n"
    "- Merge in the new step's essentials: what it accomplished (one line), key data (keep the "
    "ACTUAL numbers/dates — never vague phrasing), files it persisted (absolute paths), and its "
    "conclusion/decision.\n"
    "- DO NOT re-read, restate or elaborate older steps. Copy them forward as-is, and when you need "
    "room, COMPRESS THE OLDEST STEPS FIRST (their full data already lives in the persisted files).\n"
    "- ALWAYS retain: (a) persisted file paths, (b) the few numeric facts a later step must reuse, "
    "(c) every conclusion/decision reached.\n\n"
    "Rules:\n"
    "- Be concrete and faithful to the source; never invent data.\n"
    "- Total under ~450 words. Markdown bullet list, ordered by step number.\n"
    "- Do NOT paste raw tool JSON or long excerpts.\n"
)

_summary_cache: dict = {}


def _fallback_summary(state: dict) -> str:
    """LLM 不可用时的兜底：从 plan 状态 + artifacts + 最近一条 AI 答复提取极简要点，避免下游完全失明。"""
    plan = _normalize_plan(state.get("execution_plan") or [])
    arts = state.get("artifacts") or []
    lines = ["[auto summary — LLM unavailable, extractive fallback]"]
    for i, s in enumerate(plan):
        if s.get("status") == "completed":
            lines.append(f"- Step {i + 1} ({s.get('title', '')}): completed. {s.get('description', '')}")
    if arts:
        lines.append("- Persisted files (read with read_file for full data):")
        for p in arts[-5:]:
            lines.append(f"  - {p}")
    return "\n".join(lines)


def _obs_total(state: dict) -> int:
    """observations 的单调累计条数。

    observations 被截断到最近 40 条（从头部丢弃），len() 到达上限后不再增长，因此不能用它判断
    "本步是否产生了新产出"。这里优先取节点累计写入的 obs_total，缺失时（旧 checkpoint）回落 len()。
    """
    t = state.get("obs_total")
    try:
        return int(t) if t is not None else len(state.get("observations") or [])
    except (TypeError, ValueError):
        return len(state.get("observations") or [])


def _summarize_observations(state: dict) -> str:
    """把已完成步的 observations 提炼成"要点清单"，作为下游 agent 的跨步语义上下文。

    【增量滚动合并 —— 替换旧的全量重述实现】
    旧实现每轮都把 obs[-14:]（覆盖**全部**已完成步）重读一遍再整份重新生成，于是 step1 的数据会在
    step2/3/4/5 的 summary 里被反复复述（线上日志实证：step4 的摘要仍完整重写 step1 的三大指数点位、
    成交量、板块表现），输入随步数线性膨胀、输出随步数重复，token 与延迟都白白翻倍。
    新实现只做增量：
      - 已有摘要且无新增 observation → 直接复用，**零 LLM 调用**；
      - 首次（尚无历史摘要）→ 一次性生成全量基线；
      - 其余 → 只喂"上一步的新产出 + 现有 checklist"，要求**合并而非重述**，超预算时先压缩最旧的步。

    返回空串表示尚无已完成步（首步前）。LLM 失败时回落（保留旧摘要 + 追加新产出的提取式片段，
    绝不丢历史），不抛异常。结果按 (memory_key, obs_total, seen, 末条指纹) 缓存，避免 supervisor
    再规划与子 agent 节点在同一 state 下重复调用 LLM。同时落盘 log/<thread>_summary.md。

    设 AGENT_SUMMARY_INCREMENTAL=0 可回退旧的全量重述行为，用于对照验证。
    """
    obs = state.get("observations", []) or []
    if not obs:
        return ""
    mk = state.get("memory_key") or "default"
    total = _obs_total(state)
    seen = 0
    try:
        seen = int(state.get("summary_obs_seen") or 0)
    except (TypeError, ValueError):
        seen = 0
    prev_summary = (state.get("plan_summary") or "").strip() if _SUMMARY_INCREMENTAL else ""
    new_count = max(0, total - seen) if _SUMMARY_INCREMENTAL else total

    fp = (mk, total, seen, _msg_text(obs[-1])[:120])
    if fp in _summary_cache:
        return _summary_cache[fp]

    # 无新增产出且已有摘要 → 原样复用，零 LLM 开销（最常见：supervisor 与子节点对同一 state 重复调用）
    if prev_summary and new_count == 0:
        _summary_cache[fp] = prev_summary
        return prev_summary

    # 只取"新增"的那批 observation；计数异常（> 实际长度）时退化为最近一批，防切片越界
    new_obs = obs[-min(new_count, len(obs)):] if new_count > 0 else obs[-14:]
    arts = state.get("artifacts") or []
    art_lines = "\n".join(f"- {p}" for p in arts[-8:]) or "(none)"
    obs_parts = []
    for m in new_obs:
        if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
            c = _msg_text(m)
            who = getattr(m, "name", None) or "agent"
            obs_parts.append(f"[output from {who}] " + c[:1200])
        elif isinstance(m, ToolMessage):
            c = _msg_text(m)
            obs_parts.append(f"[tool {getattr(m, 'name', None)}] " + c[:700])
    new_text = "\n".join(obs_parts) or "(no new output captured)"
    goal = state.get("plan_goal") or _goal_text(state)

    if prev_summary:
        # 增量合并：输入只有"旧 checklist + 新一步产出"，不再重读全部历史
        sys_msg = SystemMessage(content=_SUMMARY_MERGE_PROMPT)
        user_msg = HumanMessage(content=(
            f"Overall goal: {goal}\n\n"
            f"CURRENT checklist (authoritative for older steps — copy forward, do NOT restate it):\n"
            f"{prev_summary}\n\n"
            f"NEW output from the most recently completed step (merge this in):\n{new_text}\n\n"
            f"Persisted files so far (full data lives here):\n{art_lines}\n\n"
            "Produce the UPDATED merged checklist now."
        ))
    else:
        # 首次建基线：尚无历史摘要，只能全量提炼一次（此后全部走增量）
        plan = _normalize_plan(state.get("execution_plan") or [])
        plan_lines = "\n".join(
            # 直接显示真实 status（含 failed）：若沿用 "非 completed 即 pending" 的写法，
            # 失败步会在喂给 LLM 的上下文里伪装成 pending，摘要与下游都会误判上游已就绪。
            f"{i + 1}. [{'completed' if s.get('status') == 'completed' else ('failed' if s.get('status') == 'failed' else 'pending')}] "
            f"{s.get('title', '')}: {s.get('description', '')}"
            for i, s in enumerate(plan)
        )
        sys_msg = SystemMessage(content=_SUMMARY_PROMPT)
        user_msg = HumanMessage(content=(
            f"Overall goal: {goal}\n\n"
            f"Plan status:\n{plan_lines}\n\n"
            f"Persisted files from completed steps (full data lives here):\n{art_lines}\n\n"
            f"Raw outputs of completed steps (extract key points from these; files hold full data):\n{new_text}\n\n"
            "Produce the key-points checklist now."
        ))
    text = ""
    try:
        ai = supervisor_llm.invoke([sys_msg, user_msg])
        text = ai.content if isinstance(ai, AIMessage) else str(ai)
        text = text.strip()
        if not text:
            raise ValueError("empty summary returned")
    except Exception as se:
        logger.warning(f"observations summary failed ({se}); using extractive fallback")
        # 关键：已有摘要时绝不丢历史 —— 保留旧清单，再把新产出的原始片段追加在后（截断防膨胀）
        text = _fallback_summary(state)
        if prev_summary:
            text = (
                prev_summary
                + "\n\n[auto note — checklist merge failed; new step output appended verbatim]\n"
                + new_text[:1500]
            )
    # 落盘 + 缓存
    try:
        write_summary(mk, text)
    except Exception:
        pass
    _summary_cache[fp] = text
    return text


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
      2) plan 里存在 failed 步以外的空转累积：连续 _REPLAN_NOOP_MAX 次 BEFORE==AFTER，
         说明模型对这份计划没有改写意愿，后续大概率继续空转 → 停用。
    例外（必须保留调用，否则会掩盖问题）：
      - 存在 failed 步：需要模型决定重试/改写/放弃，是再规划最有价值的场景；
      - 最后一步（current == len(plan)-1）：没有"剩余步骤"可改写，但"这一步还需不需要跑"
        正是早停（#6）要模型拍板的决策，价值高，不能省。
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
    if _REPLAN_NOOP_MAX > 0 and streak >= _REPLAN_NOOP_MAX:
        return False, f"连续 {streak} 次再规划均为空转（BEFORE==AFTER），已停用后续再规划以省 LLM 开销"
    return True, "存在待改写的剩余步骤"


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
        "  - If the completed steps have ALREADY fully achieved the goal → return next=\"FINISH\" and an "
        "'execution_plan' containing ONLY the first {n} completed steps (i.e. DROP this step).\n"
        "  - Otherwise return the agent for this step and OMIT 'execution_plan' entirely.\n"
        "Do NOT echo the plan back."
    ).format(n=current) if is_last else (
        "Decide the agent for the CURRENT step, then revise ONLY the REMAINING steps (index > current) "
        "based on what actually happened. You may skip/merge/rewrite remaining steps, but you MUST: "
        "1) keep the goal unchanged; 2) NOT increase total plan length; 3) NOT re-run completed steps; "
        "4) keep each remaining step's 'status' as 'pending' (completed steps are already marked).\n"
        "If the remaining steps need NO change, OMIT 'execution_plan' entirely — do NOT echo it back.\n"
        "EARLY FINISH: if the completed steps have already fully achieved the goal and NO remaining step "
        "is needed, return next=\"FINISH\" AND an 'execution_plan' containing ONLY the first {n} steps "
        "(i.e. drop every step from index {n} onward). A FINISH that does not truncate 'execution_plan' "
        "will be REJECTED and the step will run anyway."
    ).format(n=current)
    user_msg = HumanMessage(content=(
        f"User goal (NEVER change this):\n{goal}\n\n"
        f"Current execution_plan ('>>' marks the step being dispatched NOW, index {current}):\n{before_view}\n\n"
        f"Summary of completed steps (key points — full data is in the listed files):\n{ctx}\n\n"
        f"{revision_rule}\n\n"
        "Return strict JSON with 'next' (agent for current step or FINISH) and 'reason'."
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

    # 基线：已完成步标 completed，未完成步沿用原计划（模型不给新计划时这就是最终结果）。
    new_plan = _mark_progress(plan, current)
    revised = parsed.get("execution_plan")
    finish_reason = None
    norm_rev = _normalize_plan(revised) if isinstance(revised, list) and revised else None

    if next_agent == "FINISH":
        # 早停必须"显式截断计划"才被采信：execution_plan 长度 <= current 表明模型真的放弃了剩余步骤。
        # 只喊 FINISH 却不截断 → 驳回，仍执行当前步（防偷懒式早停导致任务半途而废）。
        if current > 0 and norm_rev is not None and 0 < len(norm_rev) <= current:
            new_plan = _mark_progress(norm_rev, len(norm_rev))
            finish_reason = str(parsed.get("reason") or "").strip() or "model judged the goal already achieved"
            logger.info(
                f"supervisor EARLY FINISH accepted after {current}/{len(plan)} steps; "
                f"remaining {len(plan) - len(norm_rev)} step(s) dropped. reason={finish_reason}"
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
# === 上下文压缩相关开关（可用 .env 覆盖，无需改代码）===
# 说明：自"跨步语义摘要（_summarize_observations）"上线后，删旧步的原始消息已不再是信息载体——
# 已完成步的"要点 + 落盘文件"由摘要承载，因此这里的截断只是最后一道防 context 溢出的安全网，
# 而非信息来源。若你担心截断漏掉重要信息，可设 AGENT_CTX_NO_TRUNCATE=1 完全关闭截断做对照验证。
_CTX_TOOL_CHARS = int(os.environ.get("AGENT_CTX_TOOL_CHARS", 1000))      # 单条 ToolMessage 保留上限
_CTX_AI_CHARS = int(os.environ.get("AGENT_CTX_AI_CHARS", 1500))          # 历史 AI 消息保留上限
_CTX_RECENT_AI_CHARS = int(os.environ.get("AGENT_CTX_RECENT_AI_CHARS", 8000))  # 最近一条上游结果上限
_CTX_KEEP_LAST = int(os.environ.get("AGENT_CTX_KEEP_LAST_MSGS", 24))     # 送入子 agent 的消息条数上限
_CTX_NO_TRUNCATE = os.environ.get("AGENT_CTX_NO_TRUNCATE", "").lower() in ("1", "true", "yes")
# 跨步语义摘要总开关（默认开启）。设 AGENT_SUMMARY_DISABLE=1 可关闭，回落"原始 observations"喂再规划。
_AGENT_SUMMARY_DISABLE = os.environ.get("AGENT_SUMMARY_DISABLE", "").lower() in ("1", "true", "yes")
# 增量滚动摘要总开关（默认开启）。设 AGENT_SUMMARY_INCREMENTAL=0 回退旧的"每轮全量重述"行为，
# 用于 A/B 对照验证：确认新实现没有丢信息、且 token 确实下降。
_SUMMARY_INCREMENTAL = os.environ.get("AGENT_SUMMARY_INCREMENTAL", "1").lower() not in ("0", "false", "no")
# 再规划（re-plan）总开关 + 空转停用阈值：
# 线上实证多轮 BEFORE==AFTER 纯回显（LLM 开销白花），连续空转达阈值后自动停用后续再规划。
_AGENT_REPLAN_DISABLE = os.environ.get("AGENT_REPLAN_DISABLE", "").lower() in ("1", "true", "yes")
_REPLAN_NOOP_MAX = int(os.environ.get("AGENT_REPLAN_NOOP_MAX", "2"))


def _msg_text(m) -> str:
    c = getattr(m, "content", "")
    return c if isinstance(c, str) else str(c)


def _truncate(text: str, limit: int) -> str:
    if _CTX_NO_TRUNCATE:
        return text
    if limit > 0 and len(text) > limit:
        return text[:limit] + f"\n...[truncated {len(text) - limit} chars; full content kept in state.observations / tmp artifact / log/<thread>_summary.md]"
    return text


def _compress_messages(msgs, keep_last: int = _CTX_KEEP_LAST):
    """pair-safe 压缩消息历史：保 首条用户 query + 最近 keep_last 条，其余按完整工具组裁剪。

    角色说明：自"跨步语义摘要"上线后，本函数只是**最后一道防 context 溢出的安全网**，不再是
    跨步信息的载体——已完成步的要点由 plan_summary 承载、全量数据由 tmp 文件承载。故：
    - 工具输出（体积最大）一律截断到 _CTX_TOOL_CHARS；
    - 历史 AI 消息截断到 _CTX_AI_CHARS 并加来源标签；最近一条（上一步的正式产出）保留更大预算；
    - 裁剪窗口时绝不切断 AIMessage(tool_calls) → ToolMessage 配对（否则部分推理服务会报
      "tool_call_id 无对应 assistant 消息"），遇到 ToolMessage 边界就后移到安全位置。
    - 设 AGENT_CTX_NO_TRUNCATE=1 可完全跳过本函数（内容截断 + 窗口裁剪都关闭），用于对照验证
      "截断是否漏掉重要信息"。
    """
    msgs = list(msgs or [])
    if not msgs:
        return msgs
    if _CTX_NO_TRUNCATE:
        return msgs  # 调试/上下文预算充足时：完全不压缩
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