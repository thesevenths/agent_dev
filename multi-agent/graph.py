"""Graph 装配 + 快照可视化 + 带记忆调用入口。

- build_graph_with_memory()：按运行模式决定 Checkpointer（直接运行挂 MemorySaver；经 LangGraph
  API 加载时不挂，由平台接管持久化）；
- visualize_snapshot(snapshot_id)：把快照渲染成 Mermaid HTML；
- invoke_with_memory(query, ...)：带记忆的图调用（供 agent.py 的 __main__ 自测与脚本调用）；
- 模块级 `graph, memory = build_graph_with_memory()`：满足 langgraph.json 的 "agent:graph" 契约
  （langgraph dev 加载 agent 模块时取 `graph` 属性）。
原定义位于 agent.py:1551-1691（含 1654 模块级 graph），拆分时整体迁入。
"""
import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage

from state import AgentState
from supervisor import supervisor
from agents import create_nodes
from planutil import members
from tools import _run_tool, list_context_snapshots, restore_snapshot

logger = logging.getLogger(__name__)


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
    nodes = create_nodes()
    for member in members:
        workflow.add_node(member, nodes[member])

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


# === 全局 Graph（满足 langgraph.json 的 "agent:graph" 契约）===
graph, memory = build_graph_with_memory()


# === 工具函数：带记忆的调用 ===
def invoke_with_memory(query: str, thread_id: str = None, config: Optional = None):
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
