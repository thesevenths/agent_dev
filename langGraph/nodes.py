import json
import logging
from typing import Annotated, Literal
from langchain_core.messages import AIMessage, HumanMessage,  SystemMessage, ToolMessage
from langgraph.types import Command, interrupt
from langchain_openai import ChatOpenAI
from state import State
from prompts import *
from tools import *


llm = ChatOpenAI(model="", temperature=0.0, base_url='', api_key='')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
hander = logging.StreamHandler()
hander.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
hander.setFormatter(formatter)
logger.addHandler(hander)

tools_map = {
    "create_file": create_file,
    "str_replace": str_replace,
    "shell_exec":  shell_exec
}

def extract_json(text):
    if '```json' not in text:
        return text
    text = text.split('```json')[1].split('```')[0].strip()
    return text

def extract_answer(text):
    if '</think>' in text:
        answer = text.split("</think>")[-1]
        return answer.strip()
    
    return text

def create_planner_node(state: State):
    logger.info("***正在运行Create Planner node***")
    messages = [SystemMessage(content=PLAN_SYSTEM_PROMPT), HumanMessage(content=PLAN_CREATE_PROMPT.format(user_message = state['user_message']))]
    response = llm.invoke(messages)
    response = response.model_dump_json(indent=4, exclude_none=True)
    response = json.loads(response)
    plan = json.loads(extract_json(extract_answer(response['content'])))
    state['messages'] += [AIMessage(content=json.dumps(plan, ensure_ascii=False))]
    return Command(goto="execute", update={"plan": plan})

def update_planner_node(state: State):
    logger.info("***正在运行Update Planner node***")
    plan = state['plan']
    goal = plan['goal']
    state['messages'].extend([SystemMessage(content=PLAN_SYSTEM_PROMPT), HumanMessage(content=UPDATE_PLAN_PROMPT.format(plan = plan, goal=goal))])
    messages = state['messages']
    while True:
        try:
            response = llm.invoke(messages)
            response = response.model_dump_json(indent=4, exclude_none=True)
            response = json.loads(response)
            plan = json.loads(extract_json(extract_answer(response['content'])))
            state['messages']+=[AIMessage(content=json.dumps(plan, ensure_ascii=False))]
            return Command(goto="execute", update={"plan": plan})
        except Exception as e:
            messages += [HumanMessage(content=f"json格式错误:{e}")]
            
def execute_node(state: State) -> Command:
    logger.info("***正在运行 execute_node***")

    plan = state["plan"]
    steps = plan["steps"]

    # 找到第一个 pending 的 step
    current_step = None
    current_step_index = 0
    for i, s in enumerate(steps):
        if s["status"] == "pending":
            current_step, current_step_index = s, i
            break

    if current_step is None or current_step_index == len(steps) - 1:
        return Command(goto="report")

    logger.info(f"当前执行 STEP: {current_step}")

    # 1. 构造本轮对话历史（不含 ToolMessage）
    messages = (
        state["observations"]
        + [SystemMessage(content=EXECUTE_SYSTEM_PROMPT)]
        + [HumanMessage(content=EXECUTION_PROMPT.format(
              user_message=state["user_message"],
              step=current_step["description"]))]
    )

    # 2. 第一次调用：让模型决定是否调用工具
    ai_msg = llm.bind_tools([create_file, str_replace, shell_exec]).invoke(messages)
    messages.append(ai_msg)          # 把带有 tool_calls 的 AIMessage 追加到历史

    # 3. 如果模型发起了工具调用，执行并追加 ToolMessage
    if ai_msg.tool_calls:
        for tc in ai_msg.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_func = tools_map[tool_name]
            tool_result = tool_func.invoke(tool_args)

            logger.info(f"tool_name:{tool_name}, tool_args:{tool_args}\ntool_result:{tool_result}")

            messages.append(ToolMessage(
                content=json.dumps(tool_result, ensure_ascii=False),
                tool_call_id=tc["id"]
            ))

        # 4. 第二次调用：让模型对工具返回结果进行总结
        final_ai_msg = llm.invoke(messages)
        messages.append(final_ai_msg)
        summary = extract_answer(final_ai_msg.content)
    else:
        summary = extract_answer(ai_msg.content)

    logger.info(f"当前 STEP 执行总结: {summary}")

    # 5. 把本轮新产生的信息写回 state
    state["messages"] += [m for m in messages if m not in state["messages"]]
    state["observations"] += [
        ToolMessage(content=json.dumps(tr, ensure_ascii=False), tool_call_id=tc["id"])
        for tc, tr in zip(ai_msg.tool_calls, [tool_result] if 'tool_result' in locals() else [])
    ]
    state["observations"] += [AIMessage(content=summary)]

    return Command(goto="update_planner", update={"plan": plan})
    

    
def report_node(state: State):
    """Report node that write a final report."""
    logger.info("***正在运行report_node***")
    
    observations = state.get("observations")
    messages = observations + [SystemMessage(content=REPORT_SYSTEM_PROMPT)]
    
    while True:
        response = llm.bind_tools([create_file, shell_exec]).invoke(messages)
        response = response.model_dump_json(indent=4, exclude_none=True)
        response = json.loads(response)
        tools = {"create_file": create_file, "shell_exec": shell_exec} 
        if response['tool_calls']:    
            for tool_call in response['tool_calls']:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                tool_result = tools[tool_name].invoke(tool_args)
                logger.info(f"tool_name:{tool_name},tool_args:{tool_args}\ntool_result:{tool_result}")
                messages += [ToolMessage(content=f"tool_name:{tool_name},tool_args:{tool_args}\ntool_result:{tool_result}", tool_call_id=tool_call['id'])]
        else:
            break
            
    return {"final_report": response['content']}



