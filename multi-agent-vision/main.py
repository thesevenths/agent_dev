from graph import app
from state import State
import json

# Example usage
initial_state = State(
    messages=[],
    user_message="帮我爬取最新的web3 加密货币新闻",  # Replace with actual user message
    plan={},
    observations=[],
    final_report=""
)

for output in app.stream(initial_state):
    for key, value in output.items():
        print(f"Output from {key}: {json.dumps(value, indent=2)}")

print("Final Report:", initial_state["final_report"])  # After run, check final_report