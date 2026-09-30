"""Agent 状态 schema —— 全局唯一真源。

agent.py 及其他模块（supervisor / plan / summary / compress / agents ...）均从此处 import
PlanStep / AgentState，不再各自定义副本。改动此处即影响全图。
"""
from typing import Annotated, Sequence, Optional, List
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
import operator


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
    # 本次 run 的启动时刻（ISO 字符串）：supervisor 规划时写入，随 checkpoint 续命。
    # 供产物幂等守卫区分"本次 run 落盘的产物"与 tmp/ 里历史 run 的同号 step 产物
    # （runlog.run_started_at 是进程内的，跨进程续跑时拿不到，故必须存 state）。
    run_started_at: Optional[str]
    # 跨会话长期记忆召回结果（run_start 每轮按当前 query 向量召回 top-k，写入此字段）：
    # supervisor 首轮规划与每个子 agent 节点都会把它作为“用户背景”注入 context（“越用越懂你”）。
    # 纯 last-write-wins 字段；为空表示本轮无相关长期记忆（或功能关闭）。详见 longterm.py。
    recalled_memory: Optional[str]
