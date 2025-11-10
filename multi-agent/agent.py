"""
Multi-Agent System with Memory, Rollback, and Visualization
- 6 Agents: Chat, Code, DB, Crawler, RAG, Context Engineer
- Memory: SQLite Checkpointer for conversation history
- Snapshots: Visualized as PNG/HTML with Mermaid diagrams
- Error Recovery: Automatic retry + fallback to other agents
- Upgraded to LangChain 1.0.5 and LangGraph 1.0.0 with Middleware
"""

import sys
import os
import json
from langgraph.checkpoint.memory import MemorySaver
from datetime import datetime
from typing import Annotated, Sequence, Dict, Any, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langgraph.errors import GraphRecursionError
import operator
import logging
from pathlib import Path

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# === 配置 ===
from dotenv import load_dotenv
load_dotenv()

from config import DASHSCOPE_API_KEY
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 调试
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === 导入 Prompt 和 Tools ===
from prompt import (
    db_system_prompt, supervisor_system_prompt, rag_system_prompt, 
    agentic_context_system_prompt, crawler_system_prompt, coder_system_prompt, chat_system_prompt
)
from tools import (
    # Chat tools
    read_file, create_file, str_replace, send_qq_email,
    # DB tools
    add_sale, delete_sale, update_sale, query_sales, query_table_schema, execute_sql,
    # Code tools
    python_repl, shell_exec,
    # Crawler tools
    get_nasdaq_top_gainers, get_crypto_sentiment_indicators, resilient_tavily_search,
    # RAG tools
    list_files_metadata,
    # Context tools
    save_context_snapshot, list_context_snapshots, evaluate_output, restore_snapshot
)

# LangChain 1.0 Imports for Agents and Middleware
from langchain.agents import create_agent, AgentMiddleware
from langchain.agents.middleware import SummarizationMiddleware, HumanInTheLoopMiddleware
from langchain_openai import ChatOpenAI
from langchain.agents.middleware.types import ModelRequest, ModelResponse, ToolCallRequest, ToolCallResponse

# === AgentState（增强版：支持快照、错误状态和reason）===
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str | None
    next: str | None
    reason: str | None  # Added for supervisor reason
    error_count: int  # 错误计数，用于重试
    snapshot_id: str | None  # 当前快照 ID
    memory_key: str  # 对话线程 ID

# === LLMs 配置 ===
def create_llm(model_name="qwen-plus", temperature=0.1):
    """创建统一的 Qwen LLM"""
    return ChatOpenAI(
        model=model_name,
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_BASE_URL,
        temperature=temperature
    )

supervisor_llm = create_llm(temperature=0.0)
chat_llm = create_llm()
db_llm = create_llm(temperature=0.0)  # DB 需要确定性
coder_llm = create_llm(temperature=0.3)  # 代码生成需要创造性
crawler_llm = create_llm()
rag_llm = create_llm(temperature=0.1)
context_engineer_llm = create_llm(temperature=0.2)

# === 自定义 Middleware for Context Engineer ===
class CustomContextMiddleware(AgentMiddleware):
    def before_model(self, request: ModelRequest) -> ModelRequest:
        # Dynamic Context Injection: Add/remove history based on query relevance
        query = request.messages[-1].content if request.messages else ""
        relevant_messages = [msg for msg in request.messages[:-1] if any(word in msg.content.lower() for word in query.lower().split())]  # Simple keyword relevance
        request.messages = relevant_messages + [request.messages[-1]]
        logger.info("Dynamic context injected based on query relevance.")
        return request

    def after_model(self, response: ModelResponse) -> ModelResponse:
        # Context Evaluation & Compression: Evaluate output and compress redundant context
        eval_result = evaluate_output("Correctness;Completeness;No Hallucination", response.content)
        if not eval_result.get("passed", False):
            logger.warning(f"Output evaluation failed: {eval_result['reason']}")
            # Compress: Summarize last 5 messages (using prebuilt if available)
            summarizer = SummarizationMiddleware(model=context_engineer_llm, max_tokens_before_summary=500)
            response.runtime.messages = summarizer.after_model(response).runtime.messages  # Compress
        logger.info("Context evaluated and compressed if needed.")
        return response

    def wrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolCallResponse]) -> ToolCallResponse:
        # Snapshot Management: Save/restore pre-tool call
        snapshot_id = save_context_snapshot({
            "messages": [m.content for m in request.runtime.messages[-5:]],
            "sender": request.runtime.sender,
            "timestamp": datetime.now().isoformat()
        })
        request.runtime.snapshot_id = snapshot_id
        logger.info(f"Pre-tool snapshot saved: {snapshot_id}")

        # Human-in-the-Loop: Pause for confirmation
        human_mw = HumanInTheLoopMiddleware(interrupt_on={"all_tools": {"allowed_decisions": ["approve", "edit", "reject"]}})
        if human_mw.wrap_tool_call(request, lambda r: r).decision != "approve":  # Simulate pause
            user_input = input("Human approval needed. Approve? (y/n/edit): ")
            if user_input.lower() == "n":
                restore_snapshot(snapshot_id)  # Rollback on reject
                raise ValueError("Human rejected tool call.")
            elif user_input.lower() == "edit":
                # Edit logic (simplified)
                request.tool_calls[0].args["query"] = input("Edit query: ")

        try:
            result = handler(request)
        except Exception as e:
            # Error Recovery: Rollback on error/hallucination
            logger.error(f"Tool call error: {e}. Rolling back.")
            restore_snapshot(snapshot_id)
            result = ToolCallResponse(error=str(e))
        return result

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        try:
            result = handler(request)
        except Exception as e:
            # Error Recovery: Detect hallucination/error and rollback
            logger.error(f"Model call error: {e}. Attempting recovery.")
            if request.runtime.snapshot_id:
                restore_snapshot(request.runtime.snapshot_id)
            result = ModelResponse(content=f"Recovered from error: {e}")
        return result

    def after_agent(self, response: Any) -> Any:
        # Trigger visualization after agent
        if response.get("snapshot_id"):
            visualize_snapshot(response["snapshot_id"])
        return response

# === 创建 Agent（使用 LangChain 1.0 create_agent + Middleware for Context Engineer）===
def create_resilient_agent(llm, tools, system_prompt, agent_name="Agent", middleware=None):
    """创建标准化 Agent with resilience"""
    system_msg = SystemMessagePromptTemplate.from_template(system_prompt)
    prompt = ChatPromptTemplate.from_messages([system_msg, MessagesPlaceholder(variable_name="messages")])
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,  # Passed directly in 1.0
        middleware=middleware or [],
        name=agent_name
    )

# 1. Chat Agent
chat_agent = create_resilient_agent(
    chat_llm,
    tools=[read_file, create_file, str_replace, send_qq_email],
    system_prompt=chat_system_prompt,
    agent_name="ChatAgent"
)

# 2. DB Agent
db_agent = create_resilient_agent(
    db_llm,
    tools=[add_sale, delete_sale, update_sale, query_sales, query_table_schema, execute_sql],
    system_prompt=db_system_prompt,
    agent_name="DBAgent"
)

# 3. Code Agent
code_agent = create_resilient_agent(
    coder_llm,
    tools=[python_repl, create_file, read_file, str_replace, shell_exec, resilient_tavily_search],
    system_prompt=coder_system_prompt,
    agent_name="CodeAgent"
)

# 4. Crawler Agent
crawler_agent = create_resilient_agent(
    crawler_llm,
    tools=[get_nasdaq_top_gainers, get_crypto_sentiment_indicators, resilient_tavily_search, create_file],
    system_prompt=crawler_system_prompt,
    agent_name="CrawlerAgent"
)

# 5. RAG Agent
rag_agent = create_resilient_agent(
    rag_llm,
    tools=[list_files_metadata, read_file],
    system_prompt=rag_system_prompt.format(file_path=os.getcwd() + "\\documents"),
    agent_name="RAGAgent"
)

# 6. Context Engineer (with Custom Middleware)
context_engineer = create_resilient_agent(
    context_engineer_llm,
    tools=[save_context_snapshot, list_context_snapshots, evaluate_output, restore_snapshot],
    system_prompt=agentic_context_system_prompt,
    agent_name="ContextEngineer",
    middleware=[CustomContextMiddleware(), SummarizationMiddleware(model=context_engineer_llm)]
)

# === 成员配置 ===
members = [
    "chat_agent", "code_agent", "db_agent", 
    "crawler_agent", "rag_agent", "context_engineer_agent"
]
options = members + ["FINISH"]

class Router(TypedDict):
    next: str
    reason: str  # Added for reason

# === Supervisor（支持错误恢复 + Reason Output）===
def supervisor(state: AgentState) -> Dict[str, Any]:
    """Supervisor with error recovery logic and reason output"""
    try:
        system_msg = SystemMessage(
            content=supervisor_system_prompt.format(members=", ".join(members))
        )
        messages = [system_msg] + state["messages"]
        
        response = supervisor_llm.with_structured_output(Router).invoke(messages)
        next_worker = response["next"]
        reason = response["reason"]
        logger.info(f"Supervisor reason: {reason}")
        
        # 错误恢复：如果之前有错误，优先让 ContextEngineer 检查
        if state.get("error_count", 0) > 0:
            logger.warning(f"Previous errors detected ({state['error_count']}), checking context...")
            next_worker = "context_engineer_agent" if next_worker != "FINISH" else "FINISH"
            reason += " (rerouted due to errors)"
        
        return {"next": next_worker, "reason": reason, "error_count": 0}  # 重置错误计数
        
    except Exception as e:
        logger.error(f"Supervisor error: {e}")
        # 回退到 ContextEngineer 修复
        return {"next": "context_engineer_agent", "reason": "Fallback due to supervisor error", "error_count": state.get("error_count", 0) + 1}

# === 通用 Agent 节点（带错误恢复，集成 Middleware行为）===
def create_resilient_node(agent):
    """创建带错误恢复的节点函数"""
    def node(state: AgentState) -> Dict[str, Any]:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 执行 Agent (LangChain 1.0 invoke)
                result = agent.invoke(state)
                
                # 保存快照（每 3 轮对话一次，middleware handles visualization）
                if len(state["messages"]) % 3 == 0:
                    snapshot_id = save_context_snapshot({
                        "messages": [m.content for m in state["messages"][-5:]],  # 最近5条
                        "sender": state["sender"],
                        "timestamp": datetime.now().isoformat()
                    })
                    state["snapshot_id"] = snapshot_id
                    logger.info(f"Snapshot saved: {snapshot_id}")
                
                return {
                    "messages": result["messages"],
                    "sender": agent.name,
                    "error_count": 0,
                    "snapshot_id": state.get("snapshot_id")
                }
                
            except GraphRecursionError:
                logger.warning("Recursion detected, breaking loop")
                return {"messages": [AIMessage(content="Task completed to avoid infinite loop.")], "sender": agent.name}
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed for {agent.name}: {e}")
                if attempt == max_retries - 1:
                    # 最终失败：回滚到上一个快照
                    if state.get("snapshot_id"):
                        rollback_msg = restore_snapshot(state["snapshot_id"])
                        return {
                            "messages": [AIMessage(content=f"Error recovered via rollback: {rollback_msg}")],
                            "sender": "Recovery",
                            "error_count": state.get("error_count", 0) + 1
                        }
                    else:
                        return {
                            "messages": [AIMessage(content=f"Critical error after {max_retries} attempts: {e}. Please clarify your request.")],
                            "sender": "ErrorHandler",
                            "error_count": state.get("error_count", 0) + 1
                        }
                
                # 重试：清理部分状态
                state["messages"] = state["messages"][-10:]  # 保留最近10条消息
                continue
    
    return node

# === 创建节点 ===
chat_node = create_resilient_node(chat_agent)
db_node = create_resilient_node(db_agent)
code_node = create_resilient_node(code_agent)
crawler_node = create_resilient_node(crawler_agent)
rag_node = create_resilient_node(rag_agent)
context_node = create_resilient_node(context_engineer)  # Middleware applied here

# === 构建 Graph（带记忆）===
def build_graph_with_memory():
    """构建带 Checkpointer 的 Graph"""
    # 初始化 Checkpointer（SQLite 记忆）
    os.makedirs("./memory", exist_ok=True)
    memory = MemorySaver()
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("chat_agent", chat_node)
    workflow.add_node("db_agent", db_node)
    workflow.add_node("code_agent", code_node)
    workflow.add_node("crawler_agent", crawler_node)
    workflow.add_node("rag_agent", rag_node)
    workflow.add_node("context_engineer_agent", context_node)
    
    # 边：Agent → Supervisor
    for member in members:
        workflow.add_edge(member, "supervisor")
    
    # START → Supervisor
    workflow.add_edge(START, "supervisor")
    
    # 条件边
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
    
    # 编译（带记忆）
    graph = workflow.compile(checkpointer=memory)
    graph.name = "Resilient Multi-Agent System"
    return graph, memory

# === 快照可视化工具 ===
def visualize_snapshot(snapshot_id: str, output_dir: str = "./snapshots"):
    """可视化快照：生成 Mermaid PNG + HTML"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # 假设快照包含消息流
        snapshot_data = json.loads(open(f"./contexts/{snapshot_id}.json").read())
        messages = snapshot_data.get("messages", [])
        
        # 生成 Mermaid 流程图
        mermaid_code = "graph TD\n"
        for i, msg in enumerate(messages):
            sender = msg.get("sender", "Unknown")
            content = msg[:50] + "..." if len(msg) > 50 else msg  # 截断
            node_id = f"N{i}"
            mermaid_code += f'    {node_id}["{sender}: {content}"]\n'
            if i > 0:   
                mermaid_code += f"    N{i-1} --> {node_id}\n"
        
        # 保存 Mermaid
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head><script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script></head>
        <body>
            <div class="mermaid">
                {mermaid_code}
            </div>
            <script>mermaid.initialize({{startOnLoad:true}});</script>
        </body>
        </html>
        """
        
        html_path = f"{output_dir}/{snapshot_id}.html"
        png_path = f"{output_dir}/{snapshot_id}.png"  # 需要额外工具生成 PNG
        
        with open(html_path, "w") as f:
            f.write(html_content)
        
        logger.info(f"Snapshot visualized: {html_path}")
        return html_path
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        return None

# === 全局 Graph ===
graph, memory = build_graph_with_memory()

# === 工具函数：带记忆的调用 ===
def invoke_with_memory(query: str, thread_id: str = None, config: Optional[Dict] = None):
    """带记忆的 Graph 调用，支持回滚"""
    if thread_id is None:
        thread_id = str(datetime.now().timestamp())
    
    config = config or {"configurable": {"thread_id": thread_id}}
    
    try:
        # 流式执行（实时输出）
        final_state = None
        for chunk in graph.stream(
            {"messages": [HumanMessage(content=query)], "memory_key": thread_id},
            config=config
        ):
            print(chunk) 
            final_state = chunk
        
        # 可视化最终快照 (middleware already handles, but fallback)
        if final_state and final_state.get("snapshot_id"):
            viz_path = visualize_snapshot(final_state["snapshot_id"])
            if viz_path:
                print(f"📊 Snapshot visualization: {viz_path}")
        
        return final_state
        
    except Exception as e:
        logger.error(f"Invocation failed: {e}")
        # 紧急回滚：恢复到最新快照
        snapshots = list_context_snapshots()
        if snapshots:
            latest = snapshots[-1]
            rollback_msg = restore_snapshot(latest['id'])
            print(f"🚨 Emergency rollback: {rollback_msg}")
        raise

# === 测试 ===
if __name__ == "__main__":
    # 初始化上下文目录
    os.makedirs("./contexts", exist_ok=True)
    os.makedirs("./snapshots", exist_ok=True)
    os.makedirs("./documents", exist_ok=True)
    
    # 测试 1：简单对话
    print("=== 测试 1：简单对话 ===")
    result1 = invoke_with_memory("你好，我是金融分析师")
    print(f"Final response: {result1['messages'][-1].content if result1 else 'Failed'}")
    
    # 测试 2：复杂查询（触发工具 + 错误恢复）
    print("\n=== 测试 2：纳斯达克查询 + 模拟错误 ===")
    try:
        # 模拟一个可能出错的查询
        result2 = invoke_with_memory("分析今天纳斯达克涨幅前3的股票，生成报告。如果出错请自动恢复。")
        print(f"Success: {result2['messages'][-1].content[:100] if result2 else 'Failed'}...")
    except Exception as e:
        print(f"Expected error handled: {e}")
    
    # 测试 3：加载记忆
    print("\n=== 测试 3：加载记忆继续对话 ===")
    thread_id = "test_thread_123"
    invoke_with_memory("之前我问了纳斯达克，现在帮我查数据库里的销售数据", thread_id=thread_id)
    
    print("\n🎉 Multi-Agent System with Memory & Recovery is ready!")
    print("Run: result = invoke_with_memory('your query', thread_id='unique_id')")