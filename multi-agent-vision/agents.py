from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from config import DASHSCOPE_API_KEY
from prompts import *
from tools import *
from state import State
import json

from dotenv import load_dotenv
import os
load_dotenv()

llm = ChatOpenAI(
    model="qwen-plus",
    temperature=0.0,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=DASHSCOPE_API_KEY
)

tools_map = {
    "create_file": create_file,
    "str_replace": str_replace,
    "shell_exec": shell_exec,
    "crawl_web3_news": crawl_web3_news,
    "query_pg_database": query_pg_database
}

def invoke_agent(messages, bound_llm):
    print(f"Bound tools: {bound_llm.tools}")  # 调试工具列表
    while True:
        ai_msg = bound_llm.invoke(messages)
        messages.append(ai_msg)
        if not ai_msg.tool_calls:
            break
        for tc in ai_msg.tool_calls:
            tool_fn = tools_map.get(tc["name"])
            tool_result = tool_fn.invoke(tc["args"]) if tool_fn else {"error": "Unknown tool"}
            messages.append(ToolMessage(content=json.dumps(tool_result), tool_call_id=tc["id"]))
    return messages[-1].content

def supervisor_agent(state: State):
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"Received user_message: {state['user_message']}")
    messages = state["messages"] + [SystemMessage(content=SUPERVISOR_PROMPT.format(user_message=state["user_message"]))]
    bound = llm.bind_tools([])  # No tools for supervisor, just routing
    response = bound.invoke(messages)
    resp = json.loads(response.content)
    state["messages"].append(AIMessage(content=response.content))
    logger.info(f"Routing to: {resp['next']}, Summary: {resp['summary']}")
    return {"next": resp["next"], "summary": resp["summary"]}

def planner_agent(state: State):
    messages = [SystemMessage(content=PLAN_SYSTEM_PROMPT)]
    if "plan" not in state or not state["plan"]:
        messages.append(HumanMessage(content=PLAN_CREATE_PROMPT.format(user_message=state["user_message"])))
    else:
        plan = state["plan"]
        messages.append(HumanMessage(content=UPDATE_PLAN_PROMPT.format(plan=plan, goal=plan["goal"])))
    bound = llm.bind(response_format={"type": "json_object"})
    content = invoke_agent(messages, bound)
    plan = json.loads(content)
    state["messages"].append(AIMessage(content=json.dumps(plan)))
    return {"plan": plan}

def data_analyst_agent(state: State):
    current_step = next((s for s in state["plan"]["steps"] if s["status"] == "pending"), None)
    if not current_step:
        return {}
    messages = state["messages"] + [
        SystemMessage(content=EXECUTE_SYSTEM_PROMPT),
        HumanMessage(content=EXECUTION_PROMPT.format(user_message=state["user_message"], step=current_step["description"]))
    ]
    bound = llm.bind_tools([create_file, str_replace, shell_exec])  # Analyst tools
    summary = invoke_agent(messages, bound)
    state["observations"].append({"type": "analysis", "content": summary})
    return state

def web_crawler_agent(state: State):
    current_step = next((s for s in state["plan"]["steps"] if s["status"] == "pending"), None)
    if not current_step:
        return {}
    messages = state["messages"] + [
        SystemMessage(content=EXECUTE_SYSTEM_PROMPT),
        HumanMessage(content=EXECUTION_PROMPT.format(user_message=state["user_message"], step=current_step["description"]))
    ]
    bound = llm.bind_tools([crawl_web3_news])
    summary = invoke_agent(messages, bound)
    state["observations"].append({"type": "crawl", "content": summary})
    return state

def database_agent(state: State):
    current_step = next((s for s in state["plan"]["steps"] if s["status"] == "pending"), None)
    if not current_step:
        return {}
    messages = state["messages"] + [
        SystemMessage(content=EXECUTE_SYSTEM_PROMPT),
        HumanMessage(content=EXECUTION_PROMPT.format(user_message=state["user_message"], step=current_step["description"]))
    ]
    bound = llm.bind_tools([query_pg_database])
    summary = invoke_agent(messages, bound)
    state["observations"].append({"type": "db", "content": summary})
    return state

def reporter_agent(state: State):
    messages = state["messages"] + [SystemMessage(content=REPORT_SYSTEM_PROMPT)]
    bound = llm.bind_tools([create_file])  # For saving report
    report = invoke_agent(messages, bound)
    return {"final_report": report}