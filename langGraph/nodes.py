import json
import logging
from typing import Annotated, Literal
from langchain_core.messages import AIMessage, HumanMessage,  SystemMessage, ToolMessage
from langgraph.types import Command, interrupt
from langchain_openai import ChatOpenAI
from state import State
from prompts import *
from tools import *

llm = ChatOpenAI(model="", temperature=0.0, base_url='', api_key='sk-')


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
    "shell_exec":  shell_exec,
    "crawl_web3_news": crawl_web3_news
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
    messages = state['messages'][:]  # 复制，避免原地改

    # 识别message中哪些tool call请求没有对应的response。 补全未解决的 tool_calls（增强匹配）
    unresolved_tool_calls = []
    for i, msg in enumerate(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                # 检查后续是否有匹配 ToolMessage
                has_response = any(
                    isinstance(m, ToolMessage) and m.tool_call_id == tc.get('id')
                    for m in messages[i+1:]
                )
                if not has_response:
                    unresolved_tool_calls.append((i, tc['id'], tc))  # 记录位置和 id

    if unresolved_tool_calls: # 插入虚拟 ToolMessage，补全message，使其满足 API 要求（每个 tool_call_id 必须有响应），从而避免 400 错误。
        logger.warning(f"Found {len(unresolved_tool_calls)} unresolved tool_calls. Inserting dummy ToolMessages at correct positions.")
        offset = 0  # 插入偏移
        for insert_idx, tool_call_id, tc in unresolved_tool_calls:
            # 创建 dummy ToolMessage
            dummy_content = json.dumps({
                "error": f"Dummy response for unresolved tool_call {tool_call_id} (original args: {tc.get('args', {})})",
                "stdout": "", "stderr": "Unresolved in planner context - original error may be driver missing"
            }, ensure_ascii=False)
            dummy_msg = ToolMessage(content=dummy_content, tool_call_id=tool_call_id)
            # 插入紧跟 AIMessage 后
            messages.insert(insert_idx + 1 + offset, dummy_msg)
            offset += 1  # 更新偏移
        state['messages'] = messages  # 更新 state

    max_retries = 3
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
                # 截断最近 AIMessage + tool_calls
                for j in range(len(messages)-1, -1, -1):
                    if isinstance(messages[j], AIMessage) and messages[j].tool_calls:
                        messages = messages[:j]  # 移除污染部分
                        break
                # messages.append(HumanMessage(content="Ignore previous tool calls and errors; directly update the plan based on available info."))
                text = (
                    "Ignore previous tool calls and errors; directly update the plan based on available info.\n\n"
                    f"{UPDATE_PLAN_PROMPT.format(plan=plan, goal=goal)}"
                )
                messages = [HumanMessage(content=text)]
            else:
                messages += [HumanMessage(content=error_msg)]
            if retry == max_retries - 1:
                raise


def execute_node(state: State) -> Command:
    logger.info("***正在运行 execute_node***")

    plan = state["plan"]
    # 容错：确保 plan 包含 steps
    if not isinstance(plan, dict) or "steps" not in plan or not isinstance(plan["steps"], list):
        logger.error(f"Invalid plan structure: {plan}. Using default steps.")
        steps = [{"title": "默认步骤", "description": "执行默认操作", "status": "pending"}]
    else:
        steps = plan["steps"]

    current_step = None
    current_step_index = 0
    for i, s in enumerate(steps):
        if s["status"] == "pending":
            current_step, current_step_index = s, i
            break

    if current_step is None or current_step_index == len(steps) - 1:
        return Command(goto="report")

    logger.info(f"当前执行 STEP: {current_step}")

    cleaned_observations = []
    for obs in state["observations"]:
        if isinstance(obs, ToolMessage):
            try:
                json.loads(obs.content)
                content = obs.content.replace('\n', '\\n').replace('\t', '\\t')
                cleaned_observations.append(ToolMessage(content=content, tool_call_id=obs.tool_call_id))
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Skipping invalid ToolMessage: {obs.content[:50]}... Error: {e}")
                continue
        else:
            cleaned_observations.append(obs)

    messages = (
            cleaned_observations
            + [SystemMessage(content=EXECUTE_SYSTEM_PROMPT)]
            + [HumanMessage(content=EXECUTION_PROMPT.format(
        user_message=state["user_message"],
        step=current_step["description"]))]
    )

    max_retries = 2
    ai_msg = None
    for retry in range(max_retries + 1):
        try:
            bound_llm = llm.bind_tools([create_file, str_replace, shell_exec, crawl_web3_news])
            ai_msg = bound_llm.invoke(messages)
            break
        except Exception as e:
            if "400" in str(e) or "Invalid parameter" in str(e):
                logger.warning(f"Retry {retry + 1}/{max_retries}: Invalid params error: {e}. Simplifying messages...")
                retry_message = f"重试执行步骤：{current_step['description']}，忽略之前的工具调用错误。"
                messages = messages[-3:] + [SystemMessage(content=EXECUTE_SYSTEM_PROMPT)] + [HumanMessage(content=retry_message)]
                if retry == max_retries:
                    raise
            else:
                raise

    messages.append(ai_msg)

    tool_results = []
    if ai_msg.tool_calls:
        logger.info(f"Tool calls detected: {[tc['name'] + ': ' + str(tc['args']) for tc in ai_msg.tool_calls]}")
        for tc in ai_msg.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_func = tools_map[tool_name]

            if tool_name == "crawl_web3_news":
                if not tool_args.get("urls") or not tool_args.get("output_file"):
                    tool_result = {"error": "urls and output_file must be non-empty"}
                else:
                    tool_result = tool_func.invoke(tool_args)
            elif tool_name == "str_replace":
                if not tool_args.get("old_str") or tool_args.get("old_str") == "" or tool_args.get("new_str") is None or tool_args.get("new_str") == "":
                    tool_result = {"error": f"Invalid args for {tool_name}: old_str/new_str empty/None"}
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
                    # 确保命令为 Windows 兼容
                    command = tool_args["command"].replace("ls", "dir", 1) if "ls" in tool_args["command"] else tool_args["command"]
                    tool_result = tool_func.invoke({"command": command})
            else:
                tool_result = tool_func.invoke(tool_args)

            tool_results.append(tool_result)
            logger.info(f"tool_name:{tool_name}, tool_args:{tool_args}\ntool_result:{tool_result}")

            messages.append(ToolMessage(
                content=json.dumps(tool_result, ensure_ascii=False),
                tool_call_id=tc["id"]
            ))

        final_ai_msg = llm.invoke(messages)
        messages.append(final_ai_msg)
        summary = extract_answer(final_ai_msg.content)
    else:
        summary = extract_answer(ai_msg.content)

    logger.info(f"当前 STEP 执行总结: {summary}")

    # 生成新 ToolMessage
    new_tool_messages = [
        ToolMessage(content=json.dumps(tr, ensure_ascii=False), tool_call_id=tc["id"])
        for tc, tr in zip(ai_msg.tool_calls or [], tool_results)
    ]

    # 新增：将所有新消息（包括 ToolMessage）添加到 state["messages"]，确保序列完整
    # 先去重：避免重复添加旧消息
    existing_ids = {id(m) for m in state["messages"]}  # 用 id() 去重（更可靠）
    new_messages_to_add = []
    for m in messages:
        if id(m) not in existing_ids:
            new_messages_to_add.append(m)
            existing_ids.add(id(m))
    state["messages"] += new_messages_to_add  # 全加：AIMessage + ToolMessage + 其他

    # observations 只加 ToolMessage 和 summary（原样）
    state["observations"] += new_tool_messages
    state["observations"] += [AIMessage(content=summary)]

    # 更新 step 状态（原样）
    plan["steps"][current_step_index]["status"] = "completed"

    return Command(goto="update_planner", update={"plan": plan})
    

    
def report_node(state: State):
    """Report node that write a final report."""
    logger.info("***正在运行report_node***")
    
    observations = state.get("observations")
    messages = observations + [SystemMessage(content=REPORT_SYSTEM_PROMPT)]
    
    while True:
        response = llm.bind_tools([create_file, shell_exec, crawl_web3_news]).invoke(messages)
        response = response.model_dump_json(indent=4, exclude_none=True)
        response = json.loads(response)
        tools = {"create_file": create_file, "shell_exec": shell_exec, "crawl_web3_news": crawl_web3_news}
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



