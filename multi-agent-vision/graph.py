from langgraph.graph import StateGraph
from agents import researcher_node, writer_node
from state import AgentState

# Define the graph
def app():
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("writer", writer_node)

    # Define edges
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "writer")
    workflow.set_finish_point("writer")

    # Compile the graph
    return workflow.compile()

# Export the graph for langgraph.json compatibility
graphs = {
    "personal-assistance": app
}