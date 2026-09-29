"""State definitions.

State is the interface between the graph and end user as well as the
data model used internally by the graph.

NOTE: The compiled multi-agent graph in agent.py defines its OWN AgentState
(TypedDict) that shadows this one — that is the authoritative schema used at
runtime. This file is kept in sync for reference / single-step tooling. If you
change one, change the other.
"""

from typing import Optional, List, Dict, Any
from typing_extensions import TypedDict
from langgraph.graph import MessagesState
from langchain_core.messages import BaseMessage
import operator
from typing import Annotated, Sequence


class PlanStep(TypedDict):
    title: str
    description: str
    status: str  # "pending" | "completed"


class AgentState(MessagesState):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: Optional[str]
    next: Optional[str]
    reason: Optional[str]
    error_count: int
    snapshot_id: Optional[str]
    memory_key: str
    hallucination_check: Optional[bool]
    # 结构化执行计划（每步带 status；对应 single-agent demo 的 Plan.steps）
    execution_plan: Optional[List[PlanStep]]
    plan_goal: Optional[str]  # 计划目标；再规划时永不改变
    observations: List  # 每步 ToolMessage + 总结 AIMessage 累积，作为再规划上下文
    current_step: int
    artifacts: List[str]  # 每步落盘产物路径（tmp/ 下），跨 agent handoff 通道
