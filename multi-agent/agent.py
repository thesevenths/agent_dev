import sys
import os
from dotenv import load_dotenv

# 调试工作目录和路径
print("Current working directory:", os.getcwd())
print("Python path:", sys.path)

# 加载环境变量
load_dotenv()
print("DASHSCOPE_API_KEY:", os.getenv("DASHSCOPE_API_KEY"))
print("TAVILY_API_KEY:", os.getenv("TAVILY_API_KEY"))

"""Define a data enrichment agent.

Works with a chat model with tool calling support.
"""

from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from tools import python_repl, add_sale, delete_sale, update_sale, query_sales, query_table_schema
from state import AgentState
from typing_extensions import TypedDict
from typing import Literal
from tools import tavily_search
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from config import DASHSCOPE_API_KEY
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# supervisor
supervisor_llm = ChatOpenAI(model="qwen-plus",
                    api_key=DASHSCOPE_API_KEY,
                    base_url=DASHSCOPE_BASE_URL)

# 用于普通问答对话
chat_llm = ChatOpenAI(model="qwen-plus",
                    api_key=DASHSCOPE_API_KEY,
                    base_url=DASHSCOPE_BASE_URL)

# 用于数据库检索
db_llm = ChatOpenAI(model="qwen-plus",
                   api_key=DASHSCOPE_API_KEY,
                   base_url=DASHSCOPE_BASE_URL)

# 用于代码生成和执行代码
coder_llm = ChatOpenAI(model="qwen-plus",
                   api_key=DASHSCOPE_API_KEY,
                   base_url=DASHSCOPE_BASE_URL)

# 爬取数据
crawler_llm = ChatOpenAI(model="qwen-plus",
                   api_key=DASHSCOPE_API_KEY,
                   base_url=DASHSCOPE_BASE_URL)


# --- 1. 创建原始 agent（不带 system prompt）---
chat_agent = create_react_agent(chat_llm, tools=[])
db_agent = create_react_agent(db_llm, tools=[add_sale, delete_sale, update_sale, query_sales, query_table_schema])
code_agent = create_react_agent(coder_llm, tools=[python_repl])
crawler_agent = create_react_agent(crawler_llm, tools=[tavily_search])

# --- 2. 定义带系统提示的节点函数 ---
def chat_agent_node(state):
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [
            SystemMessage(content="You are an intelligent chat bot.")
        ] + messages
    response = chat_agent.invoke({"messages": messages})
    return {"messages": [response["messages"][-1]], "sender": "ChatAgent"}

def db_agent_node(state):
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [
            SystemMessage(content="You perform database operations and must provide accurate data for the code_generator to use.")
        ] + messages
    response = db_agent.invoke({"messages": messages})
    return {"messages": [response["messages"][-1]], "sender": "DBAgent"}

def code_agent_node(state):
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [
            SystemMessage(content="Run Python code to display diagrams or output execution results.")
        ] + messages
    response = code_agent.invoke({"messages": messages})
    return {"messages": [response["messages"][-1]], "sender": "CodeAgent"}

def crawler_agent_node(state):
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [
            SystemMessage(content="Crawl data from the internet using search tools.")
        ] + messages
    response = crawler_agent.invoke({"messages": messages})
    return {"messages": [response["messages"][-1]], "sender": "CrawlerAgent"}


# 定义成员列表，与节点名称一致
members = ["chat_agent", "code_agent", "db_agent", "crawler_agent"]
options = members + ["FINISH"]

class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH"""
    next: Literal[*options]

def supervisor(state: AgentState):
    system_prompt = (
        f"""
        1. You are a supervisor managing a conversation between: {members}."
        2. Each has a role: chat_agent (chat), code_agent (run Python code),db_agent (database ops), crawler_agent (web search).
        3. Given the user request, choose the next worker to act. 
        4. Respond with a JSON object like {{'next': 'worker_name'}} or {{'next': 'FINISH'}}. Use JSON format strictly.
        5. know exactly when to stop the conversation and response {{'next': 'FINISH'}}.
        """
    )
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = supervisor_llm.with_structured_output(Router).invoke(messages)
    next_ = response["next"]
    # return {"next": END if next_ == "FINISH" else next_}
    return {"next": next_}   # 保持字符串，比如 "FINISH"

# --- 修复后的 workflow ---
workflow = StateGraph(AgentState)
workflow.add_node("supervisor", supervisor)
workflow.add_node("chat_agent", chat_agent_node)
workflow.add_node("db_agent", db_agent_node)
workflow.add_node("code_agent", code_agent_node)
workflow.add_node("crawler_agent", crawler_agent_node)

# 每个 agent 完成后回到 supervisor
for member in members:
    workflow.add_edge(member, "supervisor")

# 从 START 进入 supervisor
workflow.add_edge(START, "supervisor")

# supervisor 决定下一步（条件路由）
workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],
    {
        "chat_agent": "chat_agent",
        "db_agent": "db_agent",
        "code_agent": "code_agent",
        "crawler_agent": "crawler_agent",
        "FINISH": END,
    }
)

graph = workflow.compile()
graph.name = "multi-Agent"