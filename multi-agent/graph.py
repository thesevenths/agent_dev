"""Graph 装配 + 快照可视化 + 带记忆调用入口。

- build_graph_with_memory()：按运行模式决定 Checkpointer（直接运行挂 SqliteSaver→./memory/*.sqlite；
  经 LangGraph API 加载时不挂，由平台接管持久化）；
- visualize_snapshot(snapshot_id)：把快照渲染成 Mermaid HTML；
- invoke_with_memory(query, ...)：带记忆的图调用（供 agent.py 的 __main__ 自测与脚本调用）；
- 模块级 `graph, memory = build_graph_with_memory()`：满足 langgraph.json 的 "agent:graph" 契约
  （langgraph dev 加载 agent 模块时取 `graph` 属性）。
原定义位于 agent.py:1551-1691（含 1654 模块级 graph），拆分时整体迁入。
"""
import os
import json
import sqlite3
import uuid
import logging
from typing import Optional, Dict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage

from state import AgentState
from supervisor import supervisor, _INJECT_PREFIXES
from agents import create_nodes
from planutil import members
from tools import _run_tool, list_context_snapshots, restore_snapshot
from runlog import start_run, run_file, get_run_id

logger = logging.getLogger(__name__)


def run_start(state: dict) -> dict:
    """图入口节点：每次新提交（graph.stream 启动）开一个唯一 run_id 的日志文件，避免同线程复用 / 续跑时日志交织。

    仅做副作用（调用 runlog.start_run 写模块级 _current_run_id），不回写任何 state 字段，
    以免污染 AgentState 快照。返回空 dict 表示本节点不产生状态更新。
    """
    mk = state.get("memory_key") or "default"
    rid = start_run(mk)
    logger.info(f"[run_start] new run log: {run_file(rid)} (memory_key={mk})")
    # 每次新提交都打一个唯一提交序号，供 supervisor「情况0 接着聊」判定续问
    # （与平台是否在 messages 末尾注入“对话摘要”完全无关）。
    submission_id = uuid.uuid4().hex[:12]
    # 跨会话长期记忆召回（“养龙虾”闭环的读取端）：每轮入口按【当前】query（最后一条 HumanMessage）
    # 语义召回 top-k 条长期记忆，写进 state.recalled_memory，供 supervisor 首轮规划与所有子 agent
    # 节点注入。召回失败/为空/功能关闭 → 不写字段，行为与改造前完全一致（零回归）。
    try:
        from longterm import recall
        _q = ""
        for _m in reversed(list(state.get("messages") or [])):
            if isinstance(_m, HumanMessage):
                _c = _m.content if isinstance(_m.content, str) else str(_m.content)
                # 跳过本系统/平台注入的背景消息（日期上下文 / 跨会话记忆 / 上游摘要 / 线程恢复摘要），
                # 否则会拿注入背景当 query 去召回，召回质量被拖劣
                # （2026-10-07 实证：续问 run 的 recall query 竟是系统日期上下文 / 平台对话摘要，
                #  而非用户真实问题）。
                if _c.strip().startswith(_INJECT_PREFIXES):
                    continue
                _q = _c
                break
        recalled = recall(_q, memory_key=mk)
    except Exception as e:
        logger.warning(f"[run_start] long-term recall skipped ({e})")
        recalled = ""
    out = {"_submission_id": submission_id}
    if recalled:
        out["recalled_memory"] = recalled
    return out


def build_graph_with_memory():
    """构建 Graph（Checkpointer 按运行模式自动决定）

    - 直接运行（python agent.py / invoke_with_memory）：挂 MemorySaver，保留跨轮对话记忆；
    - 经 LangGraph API（langgraph dev / langgraph up）加载时，平台自带持久化，
      graph 不能带自定义 checkpointer，否则 dev 服务报 ValueError 拒绝加载。
      用 LANGSMITH_LANGGRAPH_API_VARIANT 环境变量识别，编译时不挂自定义 checkpointer。
    """
    # Checkpointer 目录：直接运行时 SqliteSaver 落这里（固定到模块目录，不随 CWD 变）；
    # langgraph dev 由平台持久化、不用它。memory 仅在直接运行分支被赋值为 SqliteSaver。
    _mem_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory")
    os.makedirs(_mem_dir, exist_ok=True)
    memory = None
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("run_start", run_start)
    workflow.add_node("supervisor", supervisor)
    nodes = create_nodes()
    for member in members:
        workflow.add_node(member, nodes[member])

    # 边：Agent → Supervisor
    for member in members:
        workflow.add_edge(member, "supervisor")

    # START → run_start → Supervisor：run_start 在每次新提交（stream 启动）时开一个
    # 唯一 run_id 的日志文件，避免同线程复用 / 续跑时日志交织到同一文件（见 runlog.py）。
    workflow.add_edge(START, "run_start")
    workflow.add_edge("run_start", "supervisor")

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
    # - 直接运行（python agent.py / invoke_with_memory）：挂 SqliteSaver，记忆落到 ./memory/*.sqlite，
    #   跨进程重启不丢（原 MemorySaver 只在内存、重启即失）；
    # - 经 LangGraph API（langgraph dev / langgraph up）加载时，平台自带持久化，
    #   graph 不能带自定义 checkpointer（否则 dev 服务报 ValueError 拒绝加载），故不挂。
    if os.environ.get("LANGSMITH_LANGGRAPH_API_VARIANT"):
        graph = workflow.compile()            # 无 checkpointer，由 LangGraph 平台接管 persistence
    else:
        # 长生命周期连接：不能用 SqliteSaver.from_conn_string（那是上下文管理器，退出即关连接）。
        # check_same_thread=False：LangGraph 流式/线程池可能跨线程访问同一连接。setup() 建表（幂等）。
        _db = os.environ.get("AGENT_CHECKPOINT_DB") or os.path.join(_mem_dir, "checkpoints.sqlite")
        memory = SqliteSaver(sqlite3.connect(_db, check_same_thread=False))
        memory.setup()
        graph = workflow.compile(checkpointer=memory)
        logger.info(f"[checkpoint] SqliteSaver → {_db}（跨重启持久化；每步自动写入，无需手动保存）")
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
# 默认会话线程：直接运行若未显式传 thread_id，用稳定值（而非每次随机时间戳），SqliteSaver 才能
# 跨进程重启命中同一条线程、把历史读回来（开箱即用的跨重启续接）。thread_id 同时用作 memory_key
# （决定 log/<key>_*.log 与 <key>_summary.md 命名）。可用 .env 的 AGENT_THREAD_ID 指定具名会话；
# 显式传参 thread_id 优先级最高。
_DEFAULT_THREAD_ID = os.environ.get("AGENT_THREAD_ID", "default")


def invoke_with_memory(query: str, thread_id: str = None, config: Optional = None):
    """带记忆的 Graph 调用，支持回滚"""
    if thread_id is None:
        thread_id = _DEFAULT_THREAD_ID

    config = config or {"configurable": {"thread_id": thread_id}}

    try:
        # 流式执行：stream_mode=["updates","messages"] 同时拿「节点级状态更新」与「逐 token 消息增量」，
        # 后者实时打印模型思考/回答的 token，缓解长等待焦虑（对齐主流 Agent 的流式体验，key_point.md #2）。
        # 多模式下每个 chunk 是 (mode, data) 元组——messages 模式 data=(message_chunk, metadata)，
        # updates 模式 data={node_name: state_update}。
        # 注：子 agent 是嵌套 invoke，其 token 仅在「父 config 被透传进子图」（HITL 开启）或
        # 经 langgraph dev/Studio 平台回调传播时才会流到这里；本自测入口至少能流外层节点消息。
        final_state = None
        for mode, data in graph.stream(
            {"messages": [HumanMessage(content=query)], "memory_key": thread_id},
            config=config,
            stream_mode=["updates", "messages"],
        ):
            if mode == "messages":
                msg_chunk = data[0] if isinstance(data, (tuple, list)) and data else None
                delta = getattr(msg_chunk, "content", "")
                if isinstance(delta, str) and delta:
                    print(delta, end="", flush=True)
            elif mode == "updates":
                for _node, upd in (data or {}).items():
                    if isinstance(upd, dict):
                        final_state = upd
        print()  # token 流结束后补一个换行

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
