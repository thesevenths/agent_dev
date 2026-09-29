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
    status: str  # "pending" | "completed" | "failed"（failed = 重试耗尽/异常，终态，不会被索引推进洗成 completed）


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
    # --- 跨步语义摘要 ---
    plan_summary: Optional[str]  # 已完成步的"要点清单"语义摘要（_summarize_observations 生成），作为下游 agent 的跨步上下文
    # --- 增量滚动摘要游标（杜绝每轮全量重述历史）---
    # 注意：observations 本身上限 40 条且从头部丢弃，故不能用 len(observations) 判断"是否有新产出"，
    # 必须另存单调计数，否则截断后 len 不再增长 → 新步产出被误判为"无新增"而整轮跳过摘要。
    obs_total: int  # 单调递增：整个运行累计追加到 observations 的条数
    summary_obs_seen: int  # 已折叠进 plan_summary 的条数（= 上次摘要时的 obs_total）
    # --- 再规划空转计数 ---
    replan_noop_streak: int  # 连续"再规划空转"（BEFORE==AFTER）次数，达阈值后停用再规划以省 LLM 开销
