from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from state import State
from nodes import (
    report_node,
    execute_node,
    create_planner_node,
    update_planner_node
)


def _build_base_graph():
    """Build and return the base state graph with all nodes and edges."""
    builder = StateGraph(State)
    builder.add_edge(START, "create_planner")
    builder.add_node("create_planner", create_planner_node)
    builder.add_node("update_planner", update_planner_node)
    builder.add_node("execute", execute_node)
    builder.add_node("report", report_node)
    builder.add_edge("report", END)
    return builder


def build_graph_with_memory():
    """Build and return the agent workflow graph with memory."""
    memory = MemorySaver()
    builder = _build_base_graph()
    return builder.compile(checkpointer=memory)


def build_graph():
    """Build and return the agent workflow graph without memory."""
    # build state graph
    builder = _build_base_graph()
    return builder.compile()


graph = build_graph()


inputs = {"user_message": "对所给文档进行分析，生成分析报告。文档使用utf-8格式读取，路径为当前目录的 E:\agent_dev\langGraph\student_habits_performance.csv",
          "plan": None,
          "observations": [],
          "final_report": ""}

# inputs = {"user_message": "我是加密货币的交易员，帮我爬取历史上加密货币的新闻，并保存到本地",
#           "plan": None,
#           "observations": [],
#           "final_report": ""}


graph.invoke(inputs, {"recursion_limit":100})