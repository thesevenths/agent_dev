import json
import logging
from typing import Annotated, Literal
from langchain_core.messages import AIMessage, HumanMessage,  SystemMessage, ToolMessage
from langgraph.types import Command, interrupt
from langchain_openai import ChatOpenAI
from state import State
from prompts import *
from tools import *


# llm = ChatOpenAI(model="", temperature=0.0, base_url='', api_key='')
llm = ChatOpenAI(model=" ", temperature=0.0, base_url=' ', api_key=' ')

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
    # response = llm.invoke(messages)
    json_llm = llm.bind(response_format={"type": "json_object"}) # 强制llm按照json格式回复
    response = json_llm.invoke(messages)
    response = response.model_dump_json(indent=4, exclude_none=True)
    response = json.loads(response)
    plan = json.loads(extract_json(extract_answer(response['content'])))
    state['messages'] += [AIMessage(content=json.dumps(plan, ensure_ascii=False))]
    return Command(goto="execute", update={"plan": plan})

def update_planner_node(state: State):
    logger.info("***正在运行Update Planner node***")
    plan = state['plan']
    goal = plan['goal']
    state['messages'].extend([SystemMessage(content=PLAN_SYSTEM_PROMPT), HumanMessage(content=UPDATE_PLAN_PROMPT.format(plan=plan, goal=goal))])
    messages = state['messages']
    
    # 新增：补全未解决的 tool_calls（核心修复）
    unresolved_tool_calls = []
    for i, msg in enumerate(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                # 检查是否有对应的 ToolMessage（简单匹配 id）
                has_response = any(
                    isinstance(m, ToolMessage) and m.tool_call_id == tc['id']
                    for m in messages[i+1:]
                )
                if not has_response:
                    unresolved_tool_calls.append((tc['id'], tc))
    
    if unresolved_tool_calls:
        logger.warning(f"Found {len(unresolved_tool_calls)} unresolved tool_calls. Adding dummy ToolMessages.")
        for tool_call_id, tc in unresolved_tool_calls:
            # 创建 dummy ToolMessage（模拟工具失败/空结果，避免 API 拒绝）
            dummy_content = json.dumps({
                "error": f"Dummy response for unresolved tool_call {tool_call_id} (original args: {tc['args']})",
                "stdout": "", "stderr": "Unresolved in planner context"
            }, ensure_ascii=False)
            messages.append(ToolMessage(content=dummy_content, tool_call_id=tool_call_id))
        # 更新 state['messages'] 以包含这些
        state['messages'] = messages  # 重新赋值
    
    max_retries = 3  # 限重试，避免无限循环
    for retry in range(max_retries):
        try:
            response = llm.invoke(messages)
            response = response.model_dump_json(indent=4, exclude_none=True)
            response = json.loads(response)
            plan = json.loads(extract_json(extract_answer(response['content'])))
            state['messages'] += [AIMessage(content=json.dumps(plan, ensure_ascii=False))]
            return Command(goto="execute", update={"plan": plan})
        except Exception as e:
            error_msg = f"json格式错误:{e}"
            logger.error(f"Retry {retry+1}/{max_retries}: {error_msg}")
            if "400" in str(e) or "invalid_request_error" in str(e):
                # 进一步简化：移除最近的 AIMessage with tool_calls（如果重试还失败）
                recent_ai_msgs = [m for m in messages[-5:] if isinstance(m, AIMessage) and m.tool_calls]
                if recent_ai_msgs:
                    messages = messages[:-len(recent_ai_msgs)]  # 截断
                    messages.append(HumanMessage(content="Ignore previous tool calls; focus on plan update."))
                else:
                    raise  # 无法修复，抛出
            else:
                messages += [HumanMessage(content=error_msg)]  # 其他错误，继续原逻辑
            if retry == max_retries - 1:
                raise  # 最终失败
            
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

    # 1. 构造本轮对话历史（不含 ToolMessage） - 新增：过滤无效 ToolMessage
    cleaned_observations = []
    for obs in state["observations"]:
        if isinstance(obs, ToolMessage):
            try:
                # 校验 content 是有效 JSON
                json.loads(obs.content)
                content = obs.content.replace('\n', '\\n').replace('\t', '\\t')  # 转义
                cleaned_observations.append(ToolMessage(content=content, tool_call_id=obs.tool_call_id))
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Skipping invalid ToolMessage: {obs.content[:50]}... Error: {e}")
                continue  # 跳过无效消息，避免污染
        else:
            cleaned_observations.append(obs)
    
    messages = (
        cleaned_observations
        + [SystemMessage(content=EXECUTE_SYSTEM_PROMPT)]
        + [HumanMessage(content=EXECUTION_PROMPT.format(
              user_message=state["user_message"],
              step=current_step["description"]))]  # 注意：step 是 dict，format 用 .format(**step) 如果需要，但这里是字符串
    )

    # 2. 第一次调用：让模型决定是否调用工具 - 新增：重试机制
    max_retries = 2
    ai_msg = None
    for retry in range(max_retries + 1):
        try:
            bound_llm = llm.bind_tools([create_file, str_replace, shell_exec])
            ai_msg = bound_llm.invoke(messages)
            break  # 成功，跳出
        except Exception as e:
            if "400" in str(e) or "Invalid parameter" in str(e):
                logger.warning(f"Retry {retry+1}/{max_retries}: Invalid params error: {e}. Simplifying messages...")
                # 简化 messages：只保留最后 3 条 + 当前 prompt，避免历史污染
                messages = messages[-3:] + [SystemMessage(content=EXECUTE_SYSTEM_PROMPT)] + [HumanMessage(content=...)]  # 替换为当前 HumanMessage
                if retry == max_retries:
                    raise  # 最终失败，抛出
            else:
                raise  # 非 400 错误，直接抛

    messages.append(ai_msg)  # 把带有 tool_calls 的 AIMessage 追加到历史

    # 3. 如果模型发起了工具调用，执行并追加 ToolMessage - 新增：预校验 args
    tool_results = []
    if ai_msg.tool_calls:
        logger.info(f"Tool calls detected: {[tc['name'] + ': ' + str(tc['args']) for tc in ai_msg.tool_calls]}")
        for tc in ai_msg.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_func = tools_map[tool_name]
            
            # 新增：args 预校验（针对常见空/None）
            if tool_name == "str_replace":
                if not tool_args.get("old_str") or tool_args.get("old_str") == "" or tool_args.get("new_str") is None or tool_args.get("new_str") == "":
                    logger.error(f"Invalid args for {tool_name}: old_str/new_str empty/None. Skipping call.")
                    tool_result = {"error": f"Invalid parameters: {tool_args} - old_str/new_str must be non-empty strings"}
                elif not tool_args.get("file_name"):
                    tool_result = {"error": "Missing file_name"}
                else:
                    tool_result = tool_func.invoke(tool_args)
            elif tool_name == "create_file":
                if not tool_args.get("file_name") or tool_args.get("file_name") == "" or tool_args.get("file_contents") is None:
                    tool_result = {"error": f"Invalid parameters: {tool_args} - file_name non-empty, file_contents required"}
                else:
                    tool_result = tool_func.invoke(tool_args)
            elif tool_name == "shell_exec":
                if not tool_args.get("command") or tool_args.get("command") == "":
                    tool_result = {"error": "Invalid parameters: command must be non-empty"}
                else:
                    tool_result = tool_func.invoke(tool_args)
            else:
                tool_result = tool_func.invoke(tool_args)
            
            tool_results.append(tool_result)
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

    # 5. 把本轮新产生的信息写回 state - 修复：只追加新 ToolMessage（用 tool_results）
    new_tool_messages = [
        ToolMessage(content=json.dumps(tr, ensure_ascii=False), tool_call_id=tc["id"])
        for tc, tr in zip(ai_msg.tool_calls or [], tool_results)
    ]
    # state["messages"] += [m for m in messages if isinstance(m, (AIMessage, HumanMessage, SystemMessage)) and m not in state["messages"]]  # 只加非 Tool
    # state["observations"] += new_tool_messages
    all_new_messages = [m for m in messages if m not in state["messages"]]  # 全加，包括 ToolMessage
    state["messages"] += all_new_messages
    state["observations"] += [AIMessage(content=summary)]

    # 更新当前 step 状态为 completed（可选，根据 summary 判断）
    plan["steps"][current_step_index]["status"] = "completed"

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



