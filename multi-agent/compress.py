"""上下文压缩（防 context 溢出的安全网，独立工具模块）。

自"跨步语义摘要"上线后，删旧步的原始消息已不再是信息载体——已完成步的"要点 + 落盘文件"
由摘要承载，故这里的截断只是最后一道防 context 溢出的安全网。设 AGENT_CTX_NO_TRUNCATE=1
可完全关闭（含窗口裁剪）做对照验证。
原定义位于 agent.py:1227-1231（开关）+ 1253-1313（函数），拆分时整体迁入。
"""
import os
import logging
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage, HumanMessage

logger = logging.getLogger(__name__)

# === 上下文压缩相关开关（可用 .env 覆盖，无需改代码）===
_CTX_TOOL_CHARS = int(os.environ.get("AGENT_CTX_TOOL_CHARS", 1000))      # 单条 ToolMessage 保留上限
_CTX_AI_CHARS = int(os.environ.get("AGENT_CTX_AI_CHARS", 1500))          # 历史 AI 消息保留上限
_CTX_RECENT_AI_CHARS = int(os.environ.get("AGENT_CTX_RECENT_AI_CHARS", 8000))  # 最近一条上游结果上限
_CTX_KEEP_LAST = int(os.environ.get("AGENT_CTX_KEEP_LAST_MSGS", 24))     # 送入子 agent 的消息条数上限
_CTX_NO_TRUNCATE = os.environ.get("AGENT_CTX_NO_TRUNCATE", "").lower() in ("1", "true", "yes")


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
