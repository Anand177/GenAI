# Refer SimpleGraph.ipynb for visualization

from langgraph.graph import StateGraph, START, END
from typing import TypedDict
import random

#Define State structure
class GraphState(TypedDict):
    max_value: int
    rand_num: int
    msg: str


# Nodes created as methods
# Random number generator
def generate_rand_num(state: GraphState) -> GraphState:
    rint = random.randint(1, state["max_value"])
    print(f"Random generated : {rint}")
    return {"rand_num": rint}

# A node for even number
def create_msg_even(state: GraphState) -> GraphState:
    return {"msg" :f"I received even number : {state['rand_num']}"}

# A node for odd number
def create_msg_odd(state: GraphState) -> GraphState:
    return {"msg" :f"I received odd number : {state['rand_num']}"}

# Conditional routing function
def route_based_on_number(state: GraphState) -> str:
    """Route to different nodes based on odd/even"""
    if state['rand_num'] % 2 ==0:
        print("Routing to create_msg_even")
        return "create_msg_even"
    else: 
        print("Routing to create_msg_odd")
        return "create_msg_odd"


# Create a graph builder object
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("generate_random", generate_rand_num)
workflow.add_node("create_msg_even", create_msg_even)
workflow.add_node("create_msg_odd", create_msg_odd)

# Define the edges between nodes
workflow.add_edge(START, "generate_random")
workflow.add_conditional_edges(
    "generate_random",  # source
    route_based_on_number,  # Routing fn
    {
        "create_msg_even": "create_msg_even",
        "create_msg_odd": "create_msg_odd"
    }
)

workflow.add_edge("create_msg_even", END)
workflow.add_edge("create_msg_odd", END)

# Compile to an executable graph
workflow_compiled = workflow.compile()


# Invoke the graph
response = workflow_compiled.invoke({"max_value" : 100})
print(response)