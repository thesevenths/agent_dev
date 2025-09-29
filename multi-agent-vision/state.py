from typing import List, TypedDict
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: List[BaseMessage]
    current_agent: str
    research_output: str
    final_output: str