from typing import Annotated, List, Dict, Literal, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class Step(TypedDict):
    title: str
    description: str
    status: Literal["pending", "completed"]

class Plan(TypedDict):
    goal: str
    thought: str
    steps: List[Step]

class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    user_message: str
    plan: Plan
    observations: List[Dict]
    final_report: str