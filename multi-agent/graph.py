"""Define a data enrichment agent.

Works with a chat model with tool calling support.
"""

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from tools import python_repl, add_sale, delete_sale, update_sale, query_sales
from state import AgentState
from typing_extensions import TypedDict
from typing import Literal
from tools import tavily_search

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


from config import DASHSCOPE_API_KEY
# 用于普通问答对话
chat_llm = ChatOpenAI(model="qwen-plus",
                    api_key=DASHSCOPE_API_KEY,
                    base_url='https://dashscope.aliyuncs.com/compatible-mode/v')

# 用于数据库检索
db_llm = ChatOpenAI(model="qwen-plus",
                   api_key=DASHSCOPE_API_KEY,
                   base_url='https://dashscope.aliyuncs.com/compatible-mode/v')

# 用于代码生成和执行代码
coder_llm = ChatOpenAI(model="qwen-plus",
                   api_key=DASHSCOPE_API_KEY,
                   base_url='https://dashscope.aliyuncs.com/compatible-mode/v')

# 爬取数据
crawler_llm = ChatOpenAI(model="qwen-plus",
                   api_key=DASHSCOPE_API_KEY,
                   base_url='https://dashscope.aliyuncs.com/compatible-mode/v')


# --- 1. 创建原始 agent（不带 system prompt）---
chat_agent = create_react_agent(chat_llm, tools=[])
db_agent = create_react_agent(db_llm, tools=[add_sale, delete_sale, update_sale, query_sales])
code_agent = create_react_agent(coder_llm, tools=[python_repl])
crawler_agent = create_react_agent(crawler_llm, tools=[tavily_search])

# --- 2. 定义带系统提示的节点函数 ---
def chat_agent_node(state):
    messages = state["messages"]
    # 插入系统消息（如果还没有）
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


# 任何一个代理都可以决定结束

members = ["chat", "coder", "sqler", "crawler"]
options = members + ["FINISH"]


class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH"""
    next: Literal[*options]


def supervisor(state: AgentState):
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {members}.\n\n"
        "Each worker has a specific role:\n"
        "- chat: Responds directly to user inputs using natural language.\n"
        "- coder: un python code to display diagrams or output execution results.\n"
        "- sqler: perform database operations while should provide accurate data for the code_generator to use.\n"
        " Given the following user request, respond with the worker to act next."
        " Each worker will perform a task and respond with their results and status."
        "When you think the result has answered the user's question, just reply FINISH."
    )

    messages = [{"role": "system", "content": system_prompt},] + state["messages"]

    response = db_llm.with_structured_output(Router).invoke(messages)
    next_ = response["next"]
    if next_ == "FINISH":
        next_ = END
    return {"next": next_}


workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor)
workflow.add_node("chat_agent", chat_agent_node)
workflow.add_node("db_agent", db_agent_node)
workflow.add_node("code_agent", code_agent_node)
workflow.add_node("crawler_agent", crawler_agent_node)

for member in members:
    # 每个子代理在完成工作后总是向主管“汇报”
    workflow.add_edge(member, "supervisor")

workflow.add_edge(START, "supervisor")
# 在图状态中填充`next`字段，路由到具体的某个节点或者结束图的运行，从来指定如何执行接下来的任务。
workflow.add_conditional_edges("supervisor", lambda state: state["next"])

# 编译图
graph = workflow.compile()

graph.name = "multi-Agent"

