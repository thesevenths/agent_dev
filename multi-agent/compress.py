"""上下文压缩（防 context 溢出，独立工具模块）。

设计分层：
  ① 主机制 = LLM 语义提炼（_llm_condense）：对"超出字符预算"的单条消息，用 LLM 按
     时间/地点/事件/要点/ID/路径/数值 的 schema 提炼，保留所有具体值（尤其 ID/URL/时间戳），
     而不是盲截断丢信息。这是用户明确要求的"提取要点、去掉冗余"。
  ② 安全网 = 盲截断（_truncate）：仅当 LLM 不可达、或 AGENT_CTX_LLM_COMPRESS=0 时启用；
     设 AGENT_CTX_NO_TRUNCATE=1 可完全关闭（含窗口裁剪）做对照验证。
  ③ 跨步摘要（summary.py 的 plan_summary）仍是已完成步的整体要点通道，与本模块互补。

原定义位于 agent.py:1227-1231（开关）+ 1253-1313（函数），拆分时整体迁入。
"""
import os
import logging
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage, HumanMessage, SystemMessage

from llm import supervisor_llm

logger = logging.getLogger(__name__)

# === 上下文压缩相关开关（可用 .env 覆盖，无需改代码）===
_CTX_TOOL_CHARS = int(os.environ.get("AGENT_CTX_TOOL_CHARS", 1000))      # 单条 ToolMessage 保留上限
_CTX_AI_CHARS = int(os.environ.get("AGENT_CTX_AI_CHARS", 1500))          # 历史 AI 消息保留上限
_CTX_RECENT_AI_CHARS = int(os.environ.get("AGENT_CTX_RECENT_AI_CHARS", 8000))  # 最近一条上游结果上限
_CTX_KEEP_LAST = int(os.environ.get("AGENT_CTX_KEEP_LAST_MSGS", 24))     # 送入子 agent 的消息条数上限
_CTX_NO_TRUNCATE = os.environ.get("AGENT_CTX_NO_TRUNCATE", "").lower() in ("1", "true", "yes")
# LLM 语义压缩总开关（默认开启）。设 AGENT_CTX_LLM_COMPRESS=0 回退纯盲截断（对照验证用）。
_CTX_LLM_COMPRESS = os.environ.get("AGENT_CTX_LLM_COMPRESS", "1").lower() not in ("0", "false", "no")

# LLM 语义提炼 schema：用户在意的"时间/地点/事件/要点/各种id号"都在保留清单里。
_CONDENSE_PROMPT = (
    "You are compressing a raw message from a multi-agent pipeline into a compact, "
    "information-preserving digest. Blind character truncation DROPS critical fields, so instead "
    "EXTRACT the decision-relevant essentials and KEEP every concrete value verbatim.\n\n"
    "Extract and preserve, in this order:\n"
    "- 时间/日期 (timestamps, dates, 'as-of' moments, trading sessions)\n"
    "- 地点/主体 (locations, entities, organizations, data sources, stock/asset names)\n"
    "- 事件 (what happened / what was found / what changed)\n"
    "- 要点 (key conclusions, decisions, risk flags)\n"
    "- 所有标识符与引用 (IDs, ticket/order numbers, entity codes, file absolute paths, URLs, "
    "numeric/stock codes, API references) — NEVER drop these even if long\n"
    "- 关键数值 (prices, volumes, percentages, counts, scores, rates)\n\n"
    "Rules:\n"
    "- Faithful to source; never invent data.\n"
    "- Keep original values verbatim where possible; do NOT paraphrase numbers/IDs/paths.\n"
    "- Output a compact markdown block; aim for ~1/3 of the original length but never omit an "
    "ID / path / URL / timestamp.\n"
    "- Use the SAME language as the source.\n"
)

# 语义提炼结果缓存（按 标签+长度+内容指纹），同一超长消息跨步复用，零重复 LLM 调用。
_condense_cache: dict = {}


def _msg_text(m) -> str:
    c = getattr(m, "content", "")
    return c if isinstance(c, str) else str(c)


def _truncate(text: str, limit: int) -> str:
    if _CTX_NO_TRUNCATE:
        return text
    if limit > 0 and len(text) > limit:
        return text[:limit] + f"\n...[truncated {len(text) - limit} chars; full content kept in state.observations / tmp artifact / log/<thread>_summary.md]"
    return text


def _llm_condense(text: str, label: str, budget: int) -> str:
    """对超过字符预算的消息做 LLM 语义提炼（保留 时间/地点/事件/要点/ID/路径/数值）。

    主路径；LLM 不可达或开关关闭时回落盲截断（_truncate），保证安全网不丢。结果按指纹缓存。
    """
    if _CTX_NO_TRUNCATE:
        return text
    # 预算内 / 关闭 LLM 压缩 → 直接盲截断（短消息无需 LLM，省开销）
    if not _CTX_LLM_COMPRESS or len(text) <= budget:
        return _truncate(text, budget)
    fp = (label, len(text), hash(text) & 0xFFFFFF)
    if fp in _condense_cache:
        return _condense_cache[fp]
    # 默认回落值：盲截断（万一 LLM 失败也至少有一份可用内容）
    condensed = _truncate(text, budget)
    try:
        ai = supervisor_llm.invoke([
            SystemMessage(content=_CONDENSE_PROMPT),
            HumanMessage(content=f"[{label}]\n{text}"),
        ])
        out = ai.content if isinstance(ai, AIMessage) else str(ai)
        out = out.strip()
        if out:
            condensed = out
    except Exception as ce:  # LLM 不可达/超时 → 不抛，回落盲截断（安全网）
        logger.warning(f"llm condense failed ({ce}); falling back to char truncation for [{label}]")
    _condense_cache[fp] = condensed
    return condensed


def _compress_messages(msgs, keep_last: int = _CTX_KEEP_LAST):
    """pair-safe 压缩消息历史：保 首条用户 query + 最近 keep_last 条，其余按完整工具组裁剪。

    角色说明：自"跨步语义摘要"上线后，本函数只是**防 context 溢出的最后一道关**，不再是跨步
    信息的唯一载体——已完成步的要点由 plan_summary 承载、全量数据由 tmp 文件承载。

    压缩策略（语义优先，截断兜底）：
    - 工具输出（体积最大、常含 ID/URL/时间戳）→ **LLM 语义提炼**（_llm_condense，保留具体值）；
    - 历史 AI 输出 → **LLM 语义提炼**（同上）；
    - 最近一条上游结果（直接 handoff）→ 保留高保真，仅盲截断到更大预算（全量已落 tmp 文件可回读）；
    - LLM 不可达 → 全部回落盲截断（安全网不丢）；设 AGENT_CTX_LLM_COMPRESS=0 可整体回退盲截断；
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
            # 工具输出体积最大且常含 ID/URL/时间戳 → LLM 语义提炼（保留具体值），盲截断仅作回落
            out.append(ToolMessage(
                content=_llm_condense(_msg_text(m), f"tool {getattr(m, 'name', None)}", _CTX_TOOL_CHARS),
                tool_call_id=getattr(m, "tool_call_id", None),
                name=getattr(m, "name", None),
            ))
        elif isinstance(m, AIMessage):
            who = getattr(m, "name", None) or "upstream agent"
            if i == last_ai_idx:
                # 最近一条上游结果 = 直接 handoff，追求高保真：仅盲截断到更大预算（全量已落 tmp 文件可回读）
                limit = _CTX_RECENT_AI_CHARS
                label = "[Most recent upstream result] "
                if not getattr(m, "tool_calls", None):
                    out.append(AIMessage(content=label + _truncate(_msg_text(m), limit), name=m.name))
                else:
                    out.append(m)
            else:
                # 历史 AI 输出 → LLM 语义提炼（保留时间/地点/事件/要点/ID），盲截断仅作回落
                limit = _CTX_AI_CHARS
                label = f"[Earlier output from {who}] "
                if not getattr(m, "tool_calls", None):
                    out.append(AIMessage(content=label + _llm_condense(_msg_text(m), f"output from {who}", limit), name=m.name))
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
