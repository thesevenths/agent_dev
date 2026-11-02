"""跨步语义摘要（要点提取）引擎。

把已完成步的 observations 提炼成"要点清单" key-points checklist，作为：
  ① supervisor 再规划上下文（比原始 observations 更准更省 token）；
  ② 下游 agent 的跨步语义上下文（与文件 handoff 共同承载信息，使原始历史可安全截断）；
  ③ 落盘 log/<thread>_summary.md 供事后回看（对应 WorkBuddy/Qoder 把关键信息提取写入 md 的行为）。

采用**增量滚动合并**：已有摘要且无新增产出 → 直接复用（零 LLM 调用）；首次 → 一次性全量
基线；其余 → 只喂"旧 checklist + 新一步产出"要求合并而非重述。LLM 不可达时回落提取式摘要。
原定义位于 agent.py:597-779，开关位于 1233/1236，拆分时整体迁入。
"""
import os
import re
import logging
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from llm import supervisor_llm
from planutil import _normalize_plan, _goal_text, _extract_json_obj
from compress import _msg_text
from runlog import write_summary

logger = logging.getLogger(__name__)

# 跨步语义摘要总开关（默认开启）。设 AGENT_SUMMARY_DISABLE=1 可关闭，回落"原始 observations"喂再规划。
_AGENT_SUMMARY_DISABLE = os.environ.get("AGENT_SUMMARY_DISABLE", "").lower() in ("1", "true", "yes")
# 增量滚动摘要总开关（默认开启）。设 AGENT_SUMMARY_INCREMENTAL=0 回退旧的"每轮全量重述"行为，
# 用于 A/B 对照验证：确认新实现没有丢信息、且 token 确实下降。
_SUMMARY_INCREMENTAL = os.environ.get("AGENT_SUMMARY_INCREMENTAL", "1").lower() not in ("0", "false", "no")

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
    "- MUST retain every concrete identifier: ticket/order/entity IDs, file absolute paths, URLs, "
    "timestamps, numeric/stock codes — never drop them even if long.\n"
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
    "- MUST retain every concrete identifier: ticket/order/entity IDs, file absolute paths, URLs, "
    "timestamps, numeric/stock codes — never drop them even if long.\n"
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


# ============================================================================
# 融合「判对错 + 摘要合并」为 1 次 LLM 调用（用户诉求：一举两得，每步省 1 次 LLM）
# ----------------------------------------------------------------------------
# 语义 critic（判本步产出是否满足要求）与跨步摘要（把本步并入 checklist）二者：
#   · 都在「一步完成时」触发；· 都读同一份新产出；· 都用 supervisor_llm。
# 故合成一次调用：模型先给 VERDICT（判定），再给 CHECKLIST（合并后的要点清单）。
# checklist 写回 state.plan_summary 后，supervisor 的 _summarize_observations 因
# new_count==0 自动短路（零 LLM），于是「判 + 压」总共只花 1 次调用。
# 判定是安全关键项，故：解析不出明确 passed → 返回 None，调用方 fail-open 回落免费规则门；
# checklist 太短/缺失 → 返回 None，调用方回落常规 _summarize_observations，绝不写坏摘要。
# ============================================================================

_FUSED_FORMAT = (
    "\n\n=== YOU MUST DO TWO THINGS IN ONE REPLY ===\n"
    "(1) QUALITY VERDICT — judge FIRST whether the NEW step output SATISFIES its STEP REQUIREMENT. "
    "Be fair but rigorous; do NOT invent extra requirements. When uncertain or only partially "
    "satisfied, prefer passed=true (never wrongly accuse). Set passed=false ONLY when it clearly "
    "fails the core ask (wrong, empty, or missing).\n"
    "(2) CHECKLIST — the updated key-points checklist per the instructions above.\n\n"
    "Reply in EXACTLY this format (VERDICT line first, then CHECKLIST):\n"
    "VERDICT: {\"passed\": true, \"reason\": \"<short and concrete>\"}\n"
    "CHECKLIST:\n<markdown bullet list>"
)


def _parse_verdict(raw: str):
    """从融合输出解析 VERDICT。返回 (passed: bool|None, reason: str)。无明确 passed → (None, '')。"""
    m = re.search(r"VERDICT\s*[:：]\s*(.*?)(?:\n\s*CHECKLIST\s*[:：]|\Z)", raw, re.DOTALL | re.IGNORECASE)
    seg = m.group(1) if m else ""
    obj = _extract_json_obj(seg) if seg.strip() else None
    if isinstance(obj, dict) and "passed" in obj:
        return bool(obj.get("passed")), str(obj.get("reason", ""))
    return None, ""


def _parse_checklist(raw: str):
    """取 CHECKLIST 段（markdown）。无该标记→剔除 VERDICT 行后的正文兜底。太短→None（调用方回落）。"""
    m = re.search(r"CHECKLIST\s*[:：]\s*(.*)\Z", raw, re.DOTALL | re.IGNORECASE)
    if m:
        cl = m.group(1).strip()
    else:
        lines = [ln for ln in raw.splitlines()
                 if not re.match(r"\s*VERDICT\s*[:：]", ln, re.IGNORECASE)]
        cl = "\n".join(lines).strip()
    # 太短的不像合格 checklist（可能模型没按格式来）→ None，交调用方回落常规摘要，避免写坏 plan_summary
    return cl if len(cl) >= 20 else None


def judge_and_summarize(step, final_text, tool_evidence, prev_summary, goal, artifacts, mk="default"):
    """融合 1 次 LLM 调用：判「本步产出是否满足要求」+ 把本步并入跨步 checklist。

    返回 (passed: bool|None, reason: str, checklist: str|None)：
      passed=None    → LLM 异常/未给明确判定 → 调用方 fail-open 回落免费规则门；
      checklist=None → 未拿到可用摘要 → 调用方回落常规 _summarize_observations。
    checklist 合格时已顺带落盘 log/<mk>_summary.md（与 _summarize_observations 对齐）。
    复用摘要的两套 prompt（增量合并 / 首次基线），叠加 VERDICT 任务；输出严格分段便于稳健解析。
    """
    desc = (f"{step.get('title', '')} {step.get('description', '')}"
            if isinstance(step, dict) else str(step))
    ev = (tool_evidence or "")[:2000]
    out_text = (final_text or "")[:4000]
    art_lines = "\n".join(f"- {p}" for p in (artifacts or [])[-8:]) or "(none)"
    base_sys = _SUMMARY_MERGE_PROMPT if prev_summary else _SUMMARY_PROMPT
    sys_msg = SystemMessage(content=base_sys + _FUSED_FORMAT)
    head = (
        f"STEP REQUIREMENT (the step just completed — judge the NEW output against THIS):\n{desc}\n\n"
        f"NEW step output (final answer):\n{out_text}\n\n"
        f"NEW step tool-output evidence:\n{ev or '(none)'}\n\n"
        f"Overall goal: {goal}\n\n"
    )
    if prev_summary:
        user_msg = HumanMessage(content=(
            head
            + f"CURRENT checklist (authoritative for older steps — copy forward, do NOT restate):\n{prev_summary}\n\n"
            + f"Persisted files so far (full data lives here):\n{art_lines}\n\n"
            + "Now reply with VERDICT then the UPDATED merged CHECKLIST."
        ))
    else:
        user_msg = HumanMessage(content=(
            head
            + f"Persisted files (full data lives here):\n{art_lines}\n\n"
            + "Now reply with VERDICT then the CHECKLIST of key points."
        ))
    try:
        ai = supervisor_llm.invoke([sys_msg, user_msg])
        raw = (ai.content if isinstance(ai, AIMessage) else str(ai)) or ""
    except Exception as ce:
        logger.warning(f"[critic+summary] fused call failed ({ce}); fail-open to rule gate")
        return None, "", None
    passed, reason = _parse_verdict(raw)
    checklist = _parse_checklist(raw)
    if checklist:
        try:
            write_summary(mk, checklist)
        except Exception:
            pass
    return passed, reason, checklist
