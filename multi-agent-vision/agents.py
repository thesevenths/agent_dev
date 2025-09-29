from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from config import Config
from prompts import RESEARCHER_PROMPT, WRITER_PROMPT
from tools import tavily_search, database_query
from langchain_core.runnables import RunnableLambda

# Initialize LLM (assuming DashScope is compatible with OpenAI's API)
llm = ChatOpenAI(
    model="gpt-4o",  # Placeholder; replace with DashScope model if needed
    api_key=Config.DASHSCOPE_API_KEY,
)

# Researcher agent node
def researcher_node(state: dict) -> dict:
    query = state["messages"][-1].content
    prompt = RESEARCHER_PROMPT.format(query=query)
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=query)
    ]
    llm_with_tools = llm.bind_tools([tavily_search, database_query])
    response = llm_with_tools.invoke(messages)
    return {
        "messages": state["messages"] + [response],
        "current_agent": "researcher",
        "research_output": response.content
    }

# Writer agent node
def writer_node(state: dict) -> dict:
    query = state["messages"][0].content
    research_output = state["research_output"]
    prompt = WRITER_PROMPT.format(query=query, research_output=research_output)
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=query)
    ]
    response = llm.invoke(messages)
    return {
        "messages": state["messages"] + [response],
        "current_agent": "writer",
        "final_output": response.content
    }