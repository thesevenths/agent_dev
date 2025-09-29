from dotenv import load_dotenv
import os

load_dotenv()  # 确保在任何代码执行前加载 .env

from langgraph.graph import END, StateGraph
from agents import supervisor_agent, planner_agent, data_analyst_agent, web_crawler_agent, database_agent, reporter_agent
from state import State

def router(state: State):
    print("Router state:", state)
    next_agent = state.get("next", "FINISH")
    if next_agent == "FINISH":
        return END
    return next_agent

graph = StateGraph(State)
graph.add_node("supervisor", supervisor_agent)
graph.add_node("planner", planner_agent)
graph.add_node("data_analyst", data_analyst_agent)
graph.add_node("web_crawler", web_crawler_agent)
graph.add_node("database", database_agent)
graph.add_node("reporter", reporter_agent)

graph.add_edge("planner", "supervisor")
graph.add_edge("data_analyst", "supervisor")
graph.add_edge("web_crawler", "supervisor")
graph.add_edge("database", "supervisor")
graph.add_edge("reporter", "supervisor")

graph.set_entry_point("supervisor")
graph.add_conditional_edges("supervisor", router, {
    "planner": "planner",
    "data_analyst": "data_analyst",
    "web_crawler": "web_crawler",
    "database": "database",
    "reporter": "reporter",
    END: END
})

app = graph.compile()

# 调试输出，确保服务器加载时可见
print("Server LANGSMITH_API_KEY:", os.getenv("LANGSMITH_API_KEY"))
print("Server LANGCHAIN_TRACING_V2:", os.getenv("LANGCHAIN_TRACING_V2"))
print("Server LANGCHAIN_PROJECT:", os.getenv("LANGCHAIN_PROJECT"))

from langsmith import Client
client = Client()
print(client.list_projects())