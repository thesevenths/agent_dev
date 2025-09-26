import json
import logging
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import Command, interrupt
from langchain_openai import ChatOpenAI
from state import State
from prompts import *
from tools import *

llm = ChatOpenAI(model="", temperature=0.0, base_url='', api_key='sk-')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

tools_map = {
    "create_file": create_file,
    "str_replace": str_replace,
    "shell_exec": shell_exec,
    "crawl_web3_news": crawl_web3_news
}


def extract_json(text: str) -> str:
    if '```json' not in text:
        return text
    text = text.split('```json')[1].split('```')[0].strip()
    return text


def extract_answer(text: str) -> str:
    if '</think>' in text:
        answer = text.split("</think>")[-1]
        return answer.strip()
    return text


def strict_pair_messages(messages: list) -> list:
    """
    严格重建 messages 序列，保证每个 ToolMessage 都紧跟在产生该 tool_call 的 AIMessage 之后，
    并且 tool_call_id 匹配。丢弃无法匹配的 ToolMessage。
    """
    result = []
    pending_tool_ids = set()
    for m in messages:
        if isinstance(m, AIMessage):
            result.append(m)
            if getattr(m, "tool_calls", None):
                # 新 AIMessage 发起了工具调用
                for tc in m.tool_calls:
                    pending_tool_ids.add(tc["id"])
        elif isinstance(m, ToolMessage):
            # 必须紧跟前一个 AIMessage 的 tool_calls 且 id 匹配
            if m.tool_call_id in pending_tool_ids:
                result.append(m)
                pending_tool_ids.remove(m.tool_call_id)
            else:
                logger.warning(f"Dropping unmatched ToolMessage id={m.tool_call_id}")
        else:
            # HumanMessage 或 SystemMessage：如果前面还有未匹配的 tool_calls，要先清空它们
            if pending_tool_ids:
                logger.warning(f"Clearing pending_tool_ids {pending_tool_ids} before inserting non-AI/non-Tool message")
                pending_tool_ids.clear()
            result.append(m)
    return result


def compress_messages_keep_ai_and_recent_sys_human(msgs, keep_recent_h=1, keep_recent_s=1):
    """
    压缩消息，只保留：
     - 所有不含 tool_calls 的 AIMessage（视为对话内容的核心历史）
     - 最近 keep_recent_h 条 HumanMessage
     - 最近 keep_recent_s 条 SystemMessage
    丢弃所有 ToolMessage 和带 tool_calls 的 AIMessage，以及曾经的 Human/System。
    """
    ai_kept = []
    last_humans = []
    last_systems = []

    # 从后往前找到最近的几条 Human / System
    for m in reversed(msgs):
        if isinstance(m, HumanMessage):
            if len(last_humans) < keep_recent_h:
                last_humans.append(m)
        elif isinstance(m, SystemMessage):
            if len(last_systems) < keep_recent_s:
                last_systems.append(m)
        # 如果两者都已够，就可以提前退出
        if len(last_humans) >= keep_recent_h and len(last_systems) >= keep_recent_s:
            break

    # 反转回来为正序
    last_humans = list(reversed(last_humans))
    last_systems = list(reversed(last_systems))

    # 遍历原消息，挑出 AIMessage（无 tool_calls）
    for m in msgs:
        if isinstance(m, AIMessage):
            tc = getattr(m, "tool_calls", None)
            if not tc:
                ai_kept.append(m)

    # 组合：AI 历史 + 最近 System + 最近 Human
    compressed = []
    compressed.extend(ai_kept)
    compressed.extend(last_systems)
    compressed.extend(last_humans)
    return compressed

def invoke_with_tools(messages: list):
    """
    调用 LLM（绑定工具）直到没有 tool_calls 为止：
    AI→ToolMessage→AI→ … 交替循环。
    返回最终的 messages（含 AI、ToolMessage）和最后的 AIMessage。
    """
    bound = llm.bind_tools(list(tools_map.values()))
    # ensure current messages are strictly paired
    messages = strict_pair_messages(messages)
    while True:
        ai_msg = bound.invoke(messages)
        messages.append(ai_msg)
        if not getattr(ai_msg, "tool_calls", None):
            break
        # 有 tool_calls，则执行所有 tool，插入对应 ToolMessages，然后继续
        for tc in ai_msg.tool_calls:
            tool_name = tc["name"]
            args = tc["args"]
            tool_fn = tools_map.get(tool_name)
            if tool_fn is None:
                tool_result = {"error": f"Unknown tool: {tool_name}"}
            else:
                try:
                    tool_result = tool_fn.invoke(args)
                except Exception as e:
                    tool_result = {"error": str(e)}
            tm = ToolMessage(content=json.dumps(tool_result, ensure_ascii=False),
                             tool_call_id=tc["id"])
            messages.append(tm)
        # loop 继续，下一轮 AI 可能基于这些 ToolMessage 继续
    return messages, ai_msg


def create_planner_node(state: State):
    logger.info("*** 正在运行 create_planner_node ***")
    messages = [
        SystemMessage(content=PLAN_SYSTEM_PROMPT),
        HumanMessage(content=PLAN_CREATE_PROMPT.format(user_message=state["user_message"]))
    ]
    # 强制 LLM 返回 JSON
    json_llm = llm.bind(response_format={"type": "json_object"})
    ai_response = json_llm.invoke(messages)
    # ai_response 可能自带 tool_calls，也可能没有；但在这个阶段通常无工具调用
    # 转为 dict
    resp_json = ai_response.model_dump_json(indent=4, exclude_none=True)
    resp_obj = json.loads(resp_json)
    plan = json.loads(extract_json(extract_answer(resp_obj["content"])))
    state["messages"].append(AIMessage(content=json.dumps(plan, ensure_ascii=False)))
    return Command(goto="execute", update={"plan": plan})


def update_planner_node(state: State):
    logger.info("*** 正在运行 update_planner_node ***")
    plan = state["plan"]
    goal = plan.get("goal")
    # state["messages"].extend([
    #     SystemMessage(content=PLAN_SYSTEM_PROMPT),
    #     HumanMessage(content=UPDATE_PLAN_PROMPT.format(plan=plan, goal=goal))
    # ])
    # messages = state["messages"][:]
    # messages = state["messages"][:]
    # messages = state["observations"][:]  # 这里可以用state["observations"][:]替代，减少message
    # 压缩历史消息
    messages = compress_messages_keep_ai_and_recent_sys_human(state["messages"], keep_recent_h=1, keep_recent_s=1)

    # 再加上本次的 System + Human prompt
    messages.append(SystemMessage(content=PLAN_SYSTEM_PROMPT))
    messages.append(HumanMessage(content=UPDATE_PLAN_PROMPT.format(plan=plan, goal=goal)))

    max_retries = 3
    for retry in range(max_retries):
        try:
            messages = strict_pair_messages(messages)
            ai_msg = llm.invoke(messages)
            messages.append(ai_msg)
            resp_json = ai_msg.model_dump_json(indent=4, exclude_none=True)
            resp_obj = json.loads(resp_json)
            plan = json.loads(extract_json(extract_answer(resp_obj["content"])))
            state["messages"].append(AIMessage(content=json.dumps(plan, ensure_ascii=False)))
            return Command(goto="execute", update={"plan": plan})
        except Exception as e:
            logger.error(f"Retry {retry+1}/{max_retries} — JSON parse / invoke error: {e}")
            # 发生错误就截断最近的 AIMessage + tool_calls 部分，重构 messages
            # 保留历史消息至最近能用的位置
            trimmed = []
            for m in reversed(messages):
                if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                    break
                trimmed.append(m)
            trimmed = list(reversed(trimmed))
            messages = trimmed + [
                HumanMessage(content="忽略之前错误的调用，重新尝试更新计划。\n"
                                     f"{UPDATE_PLAN_PROMPT.format(plan=plan, goal=goal)}")
            ]
            if retry == max_retries - 1:
                raise
    # 若都失败
    raise RuntimeError("update_planner_node: exceeded retry limit")


def execute_node(state: State) -> Command:
    logger.info("*** 正在运行 execute_node ***")
    plan = state.get("plan", {})
    steps = plan.get("steps")
    if not isinstance(steps, list):
        logger.error(f"Invalid plan.steps: {steps}, fallback to default")
        steps = [{"title": "默认","description": "执行默认", "status": "pending"}]
    current_index = None
    for idx, step in enumerate(steps):
        if step.get("status") == "pending":
            current_index = idx
            break
    if current_index is None:
        return Command(goto="report")
    current_step = steps[current_index]
    desc = current_step.get("description", "")

    # 合并 observations + system + human prompt
    messages = state.get("observations", [])[:]  # copy
    messages.append(SystemMessage(content=EXECUTE_SYSTEM_PROMPT))
    messages.append(HumanMessage(content=EXECUTION_PROMPT.format(
        user_message=state["user_message"],
        step=desc
    )))

    # 调用，自动走 AI→ToolMessage→AI 的流程
    messages, last_ai = invoke_with_tools(messages)

    summary = extract_answer(last_ai.content)
    logger.info(f"Step 执行 summary: {summary}")

    # 将新 messages 加入 state["messages"]
    state["messages"].extend(messages)
    # 仅 observations 保留 ToolMessage + summary AIMessage
    for m in messages:
        if isinstance(m, ToolMessage):
            state["observations"].append(m)
    state["observations"].append(AIMessage(content=summary))

    # 标记该 step 为完成
    plan["steps"][current_index]["status"] = "completed"
    return Command(goto="update_planner", update={"plan": plan})


def report_node(state: State):
    logger.info("*** 正在运行 report_node ***")
    observations = state.get("observations", [])
    messages = strict_pair_messages(observations[:]) + [SystemMessage(content=REPORT_SYSTEM_PROMPT)]
    # 最后调用一次，不要让 ToolMessage 插入出错顺序
    messages, last_ai = invoke_with_tools(messages)
    # last_ai.content 就是最终报告
    return {"final_report": last_ai.content}
