"""人机协同确认（Human-in-the-Loop, HITL）—— 工具风险分级 + 危险工具调用前中断确认。

设计（对应 key_point.md #1）：
  - 工具风险分级注册表 TOOL_RISK：
      safe   只读/查询（read_file / grep_files / query_* / get_*），直接放行；
      warn   写入/改数据/跑代码（create_file / str_replace / python_repl / add_sale / update_sale），
             有副作用但高频，仅靠 ToolCallLoggingMiddleware 记录，不阻断；
      danger 不可逆或强外部副作用（send_qq_email / delete_sale / execute_sql / shell_exec），
             调用前必须人类确认 → 触发 interrupt()。
  - 仅 danger 级触发中断：子 agent 的 LLM 决定调用危险工具时，HumanInTheLoopMiddleware 的
    after_model 钩子暂停执行，把「将要执行的工具 + 参数」抛给人类；在 LangGraph Studio 里
    选择 approve（照常执行）/ edit（改参数后执行）/ reject（拒绝，让模型改走别的路）后才继续。
  - 总开关 AGENT_HITL_ENABLED（默认关）：关闭时不挂中间件、不中断、不透传父 config，
    行为与改造前 100% 一致（零回归）。危险工具集可用 AGENT_HITL_DANGER_TOOLS 逗号覆盖。

依赖：仅 langchain.agents.middleware（叶子模块，不 import 本项目其他模块，避免循环依赖）。

生效前提（开启后）：
  1) 外层图必须带 checkpointer 才能暂停/恢复 —— langgraph dev 由平台托管；invoke_with_memory
     挂 MemorySaver（见 graph.build_graph_with_memory）；
  2) 节点必须把父 config 透传进子 agent.invoke（见 agents.create_resilient_node），否则子图内的
     interrupt() 无法冒泡到外层图、也无法被恢复。
"""
import os
import json
import logging

from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents.middleware.human_in_the_loop import InterruptOnConfig

logger = logging.getLogger(__name__)

# === 总开关（默认关：不挂中间件、不中断，行为与改造前完全一致）===
HITL_ENABLED = os.environ.get("AGENT_HITL_ENABLED", "0").strip().lower() in ("1", "true", "yes", "on")

# === 工具风险分级（单一真源）===
TOOL_RISK = {
    # danger：不可逆 / 强外部副作用 —— 调用前必须人类确认
    "send_qq_email": "danger",
    "delete_sale": "danger",
    "execute_sql": "danger",
    "shell_exec": "danger",
    # warn：有副作用但高频 —— 仅记录，不阻断（避免每步都被打断）
    "add_sale": "warn",
    "update_sale": "warn",
    "python_repl": "warn",
    "create_file": "warn",
    "str_replace": "warn",
    # 其余未列出的工具默认按 safe 处理（只读/查询）
}

_DANGER_DEFAULT = [t for t, r in TOOL_RISK.items() if r == "danger"]
_env_danger = os.environ.get("AGENT_HITL_DANGER_TOOLS", "")
DANGER_TOOLS = (
    {t.strip() for t in _env_danger.split(",") if t.strip()}
    if _env_danger.strip() else set(_DANGER_DEFAULT)
)


def risk_of(tool_name: str) -> str:
    """返回某工具的风险级别：safe / warn / danger（未登记默认 safe）。"""
    return TOOL_RISK.get(tool_name, "safe")


def _describe(tool_call, state, runtime) -> str:
    """中断时展示给人类的「将要执行什么」描述（工具名 + 参数 + 风险说明 + 决策项）。"""
    try:
        args = json.dumps(tool_call.get("args", {}), ensure_ascii=False, default=str)
    except Exception:
        args = str(tool_call.get("args"))
    return (
        "⚠️ 高风险操作待确认（HITL）\n"
        f"工具: {tool_call.get('name')}\n"
        f"参数: {args}\n"
        "该操作不可逆或有强外部副作用（发邮件 / 删数据 / 任意 SQL / 任意 shell 命令）。\n"
        "approve=照常执行 | edit=改参数后执行 | reject=拒绝并让模型改走别的路。"
    )


def build_interrupt_on(tool_names) -> dict:
    """给定某 agent 绑定的工具名集合，返回 {tool_name: InterruptOnConfig}（仅含 danger 级工具）。

    只有登记为危险、且该 agent 实际绑定了的工具才会进入 interrupt_on；空 dict 表示无需中断。
    """
    cfg = {}
    for name in tool_names:
        if name and name in DANGER_TOOLS:
            cfg[name] = InterruptOnConfig(
                allowed_decisions=["approve", "edit", "reject"],
                description=_describe,
            )
    return cfg


def hitl_middleware(tool_names):
    """若 HITL 开启且该 agent 含危险工具，返回 [HumanInTheLoopMiddleware]，否则返回 []。

    以「该 agent 实际绑定的工具」过滤，避免给不含危险工具的 agent 挂无用中间件。
    """
    if not HITL_ENABLED:
        return []
    interrupt_on = build_interrupt_on(tool_names)
    if not interrupt_on:
        return []
    logger.info(f"[hitl] gating danger tool(s): {sorted(interrupt_on)}")
    return [HumanInTheLoopMiddleware(
        interrupt_on=interrupt_on,
        description_prefix="高风险工具调用需人工确认",
    )]
