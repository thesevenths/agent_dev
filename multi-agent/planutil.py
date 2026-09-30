"""计划解析 / 成员路由配置（叶子模块，无内部依赖）。

集中存放：
  - members / options：图里 6 个 agent 节点名（含 FINISH），供 _parse_target_agent、
    supervisor 再规划 prompt、build_graph 条件边共用，单一真源；
  - Router：supervisor 结构化输出 schema；
  - _extract_json_obj / _normalize_step / _normalize_plan / _parse_target_agent /
    _goal_text / _latest_result_text / _build_replan_context：agent.py 里原"计划解析
    与 introspection"一组纯函数，仅依赖 langchain 消息类型与标准库，不触达 LLM / summary，
    故抽成叶子以避免 summary↔plan 循环依赖。
"""
from typing import Optional, List
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import re
import json
import os

from state import PlanStep

# 结构化输出 parse 失败时的“带错误重试”次数（把上一次报错喂回模型让其重生成）。
_PARSE_TRIES = int(os.environ.get("AGENT_PARSE_TRIES", "3"))


# === 成员配置（图节点名，单一真源）===
members = [
    "chat_agent", "code_agent", "db_agent",
    "crawler_agent", "rag_agent", "context_engineer_agent",
]
options = members + ["FINISH"]


class Router(TypedDict):
    next: str
    reason: str  # Added for reason
    execution_plan: Optional[List[PlanStep]]   # 首次规划或再规划时输出（结构化步骤列表）
    goal: Optional[str]  # 仅首次规划时输出目标


def _structured_with_retry(llm, messages, schema, tries: int | None = None, label: str = "structured"):
    """结构化输出 + parse 失败带错误重试（鲁棒性#3）。

    首次失败不再直接降级，而是把上一次的 schema/parse 报错作为一条 HumanMessage 喂回模型，
    让其仅重生成合法 JSON，最多 tries 次（默认 AGENT_PARSE_TRIES=3）；全部失败返回 None，
    由调用方回落“裸调用 + 手动抽 JSON”。llm 以参数传入，保持本模块叶子属性（不 import llm）。
    """
    n = tries if tries is not None else _PARSE_TRIES
    msgs = list(messages)
    last_err: object = None
    for attempt in range(max(1, n)):
        try:
            resp = llm.with_structured_output(schema).invoke(msgs)
            if isinstance(resp, dict):
                return resp
            last_err = f"non-dict result: {type(resp).__name__}"
        except Exception as e:  # schema 校验/网络/后端异常都算一次失败
            last_err = e
        if attempt < n - 1:
            msgs = list(messages) + [HumanMessage(content=(
                f"Your previous reply failed schema validation / JSON parsing: {last_err}. "
                "Return ONLY a single strict, valid JSON object matching the required schema now. "
                "Do not add prose or code fences."
            ))]
    import logging as _lg
    _lg.getLogger(__name__).warning(
        f"{label}: structured output failed after {n} tries ({last_err}); caller will fall back"
    )
    return None


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
