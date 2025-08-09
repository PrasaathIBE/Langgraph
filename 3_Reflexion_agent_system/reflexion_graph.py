from typing import List 
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END, MessageGraph
from chains import revisor_chain, first_responder_chain
from execute_tools import execute_tools

graph = MessageGraph()
MAX_ITERATIONS = 2

# Define the graph structure
graph.add_node("draft", first_responder_chain)
graph.add_node("exexute_tools", execute_tools)
graph.add_node("revisor", revisor_chain)

graph.add_edge("draft", "exexute_tools")
graph.add_edge("exexute_tools", "revisor")


# Define the event loop 
def event_loop(state: List[BaseMessage]) -> str:
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
    num_iterations = count_tool_visits
    if num_iterations > MAX_ITERATIONS:
        return END
    return "exexute_tools"

graph.add_conditional_edges("revisor", event_loop)
graph.set_entry_point("draft")

app = graph.compile()

print(app.get_graph().draw_mermaid())

response = app.invoke(
    "Write about how small businesses can leverage AI to grow"
)

# Get the last message in the response
final_message = response[-1]

# Extract the answer from the last tool call
final_tool_call = final_message.tool_calls[-1]
final_answer = final_tool_call['args']['answer']

print(final_answer)