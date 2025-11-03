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
from tools import get_nasdaq_top_gainers, python_repl, add_sale, delete_sale, update_sale, query_sales, query_table_schema, execute_sql, create_file, str_replace, shell_exec, list_files_metadata, read_file
from state import AgentState
from typing_extensions import TypedDict
from typing import Literal
from tools import tavily_search
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from prompt import db_system_prompt, supervisor_system_prompt, rag_system_prompt, agentic_context_system_prompt, crawler_system_prompt, coder_system_prompt
from tools import save_context_snapshot, list_context_snapshots, evaluate_output

from config import DASHSCOPE_API_KEY

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# print(DASHSCOPE_BASE_URL)
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


# RAG agent：读取知识库回答问题
rag_llm = ChatOpenAI(model="qwen-plus",
                      api_key=DASHSCOPE_API_KEY,
                      base_url=DASHSCOPE_BASE_URL)

# 负责规划/保存快照/验证/回滚等
context_engineer_llm = ChatOpenAI(model="qwen-plus",
                                  api_key=DASHSCOPE_API_KEY,
                                  base_url=DASHSCOPE_BASE_URL)


# --- 1. 创建原始 agent（不带 system prompt）---
chat_agent = create_react_agent(chat_llm, tools=[])

db_agent = create_react_agent(
    model=db_llm,  
    tools=[add_sale, delete_sale, update_sale, query_sales, query_table_schema, execute_sql],
    prompt=ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(db_system_prompt),
        MessagesPlaceholder(variable_name="messages"), 
    ])
)

code_agent = create_react_agent(
    coder_llm, 
    tools=[python_repl, create_file, str_replace, shell_exec],
     prompt=ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(coder_system_prompt),
        MessagesPlaceholder(variable_name="messages"), 
    ])
)

crawler_agent = create_react_agent(
    crawler_llm, 
    tools=[get_nasdaq_top_gainers, tavily_search],
     prompt=ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(crawler_system_prompt),
        MessagesPlaceholder(variable_name="messages"), 
    ])
)

rag_agent = create_react_agent(
    rag_llm,
    tools=[list_files_metadata, read_file],
    prompt=ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(rag_system_prompt.format(file_path=os.getcwd() + "\\documents")),
        MessagesPlaceholder(variable_name="messages"),  
    ])
)

context_engineer = create_react_agent(
    context_engineer_llm,
    tools=[save_context_snapshot, list_context_snapshots, evaluate_output],
    prompt=ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(agentic_context_system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])
)

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


def rag_agent_node(state):
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [
                       SystemMessage(content="You are an agentic retrieval-augmented generation (RAG) agent.")
                   ] + messages
    response = rag_agent.invoke({"messages": messages})
    return {"messages": [response["messages"][-1]], "sender": "RAGAgent"}


def context_engineer_agent_node(state):
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [
            SystemMessage(content="You are a Context Engineer: plan, patch, verify, snapshot, rollback if needed.")
        ] + messages
    response = context_engineer.invoke({"messages": messages})
    return {"messages": [response["messages"][-1]], "sender": "ContextEngineer"}


# 定义成员列表，与节点名称一致
members = ["chat_agent", "code_agent", "db_agent", "crawler_agent", "rag_agent", "context_engineer_agent"]
options = members + ["FINISH"]


class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH"""
    next: Literal[*options]


def supervisor(state: AgentState):
    # system_prompt = (
    #     f"""
    #     1. You are a supervisor managing a conversation between: {members}."
    #     2. Each has a role: chat_agent (chat), code_agent (run Python code),db_agent (database ops), crawler_agent (web search).
    #     3. Given the user request, choose the next worker to act.
    #     4. Respond with a JSON object like {{'next': 'worker_name'}} or {{'next': 'FINISH'}}. Use JSON format strictly.
    #     5. know exactly when to stop the conversation and response {{'next': 'FINISH'}}.
    #     """
    # )

    # print("🔍 Supervisor called!")
    # print("DASHSCOPE_API_KEY (from env):", os.getenv("DASHSCOPE_API_KEY"))
    # print("DASHSCOPE_BASE_URL:", repr(DASHSCOPE_BASE_URL))  # 注意 repr 能看到空格！

    messages = [SystemMessage(content=supervisor_system_prompt.format(members=members))] + state["messages"]
    response = supervisor_llm.with_structured_output(Router).invoke(messages)
    next_ = response["next"]
    # return {"next": END if next_ == "FINISH" else next_}
    return {"next": next_}  # 保持字符串，比如 "FINISH"


# --- 修复后的 workflow ---
workflow = StateGraph(AgentState)
workflow.add_node("supervisor", supervisor)
workflow.add_node("chat_agent", chat_agent_node)
workflow.add_node("db_agent", db_agent_node)
workflow.add_node("code_agent", code_agent_node)
workflow.add_node("crawler_agent", crawler_agent_node)
workflow.add_node("rag_agent", rag_agent_node)
workflow.add_node("context_engineer_agent", context_engineer_agent_node)

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
        "rag_agent": "rag_agent",
        "context_engineer_agent": "context_engineer_agent",
        "FINISH": END,
    }
)

graph = workflow.compile()
graph.name = "multi-Agent"

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    result = graph.invoke({
        "messages": [HumanMessage(content="你好")]
    })
    print(result)

"""
    todo:
        - Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models
        - https://www.arxiv.org/pdf/2510.04618  
        - https://mp.weixin.qq.com/s/f-1h0Q-QKOWghJb7Fmrvtw   context adaptation
"""