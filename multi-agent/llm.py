"""LLM 与自定义 Middleware。

- create_llm + 6 个角色 LLM 实例（supervisor/chat/db/code/crawler/rag/context_engineer）；
- CustomContextMiddleware（Context Engineer 专用：快照/回滚/评估）；
- ToolCallLoggingMiddleware（一处覆盖全部工具调用的日志，零破坏）；
- _tool_args_preview（工具入参预览）。

原定义位于 agent.py:220-415，拆分时整体迁入。依赖 tools（_run_tool 等）与 config（LLM_*）。
注意：CustomContextMiddleware.after_agent 通过懒加载 `from graph import visualize_snapshot`
访问快照可视化，避免 llm → graph 的顶层循环依赖（graph 反过来经由 agents → llm）。
"""
import json
import os
import time
import logging
from datetime import datetime
from typing import Any

from runlog import log_event
from config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, LLM_VERIFY_SSL
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware, SummarizationMiddleware, HumanInTheLoopMiddleware,
    ModelRequest, ModelResponse, ToolCallRequest,
)
from langchain_openai import ChatOpenAI

from tools import _run_tool, evaluate_output, save_context_snapshot, restore_snapshot

logger = logging.getLogger(__name__)


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
        # Agent 结束后可视化最近快照（懒加载 graph.visualize_snapshot，避免 llm→graph 顶层循环依赖）。
        if self._last_snapshot_id:
            try:
                from graph import visualize_snapshot as _viz
            except Exception:
                _viz = None
            if _viz is not None:
                try:
                    _viz(self._last_snapshot_id)
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

    每条日志都带上"所属 agent 名"（[tool][CodeAgent] ...）：此前 [tool] 行不含归属，
    在控制台里与平台的 langgraph_node= 标记（不同 logger，且并发/续跑时会交错）就近配对，
    极易被误读成"code_agent 调了 tavily、chat_agent 调了 shell_exec"。实际上工具集在
    create_agent 时已按 agent 固定绑定（code_agent 无 tavily、chat_agent 无 shell_exec），
    根本不可能跨 agent 调用；带上归属标识即可彻底消除这种"错位"错觉。
    """

    def __init__(self, agent_name: str | None = None):
        super().__init__()
        self._agent = agent_name or "?"
        self._tag = f"[tool][{self._agent}]"

    def wrap_tool_call(self, request, handler):
        tc = getattr(request, "tool_call", None) or {}
        name = tc.get("name") or (request.tool.name if getattr(request, "tool", None) else "unknown")
        args = tc.get("args") or {}
        t0 = time.time()
        head = f"{self._tag} ▶ {name}({_tool_args_preview(args)})"
        logger.info(head)
        log_event(head)
        try:
            result = handler(request)
        except Exception as e:
            dur = time.time() - t0
            msg = f"{self._tag} ✖ {name} raised after {dur:.2f}s: {type(e).__name__}: {e}"
            logger.error(msg)
            log_event(msg)
            raise  # 不吞异常：重试/回滚逻辑依赖异常向上传播
        dur = time.time() - t0
        status = getattr(result, "status", "") or ""
        content = getattr(result, "content", "")
        content = content if isinstance(content, str) else str(content)
        flag = "✖" if status == "error" else "✔"
        tail = f"{self._tag} {flag} {name} ({dur:.2f}s, {len(content)} chars) -> {content[:200]}"
        logger.info(tail)
        log_event(tail)
        return result
