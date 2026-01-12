# Refer SimpleGraph2.ipynb for visualization
from langgraph.graph import StateGraph, START, END
from sympy import *
from typing import List,TypedDict
import random

#Define State structure
class GraphState(TypedDict):
    max_value: int
    rand_num: int
    msg: str
    rand_num_list : List[int]


# Nodes created as methods
# Random number generator
def generate_rand_num(state: GraphState) -> GraphState:
    rint = random.randint(1, state["max_value"])
    print(f"Random generated : {rint}")
    rand_num_lst = state["rand_num_list"]
    rand_num_lst.append(rint)
    return {"rand_num": rint, "rand_num_list" : rand_num_lst}

# A node for even number
def create_msg_even(state: GraphState) -> GraphState:
    return {"msg" :f"I received even number : {state['rand_num']}"}

# A node for odd number
def create_msg_odd(state: GraphState) -> GraphState:
    return {"msg" :f"I received odd number : {state['rand_num']}"}

# A node for 4 multiple number
def create_msg_4(state: GraphState) -> GraphState:
    return {"msg" :f"I received multiple of 4 : {state['rand_num']}"}

# A node for even number
def create_msg_prime(state: GraphState) -> GraphState:
    return {"msg" :f"I received prime number : {state['rand_num']}"}

# Conditional routing function
def route_based_on_number(state: GraphState) -> str:
    """Route to different nodes based on odd/even"""
    if state['rand_num'] % 2 ==0:
        print("Routing to create_msg_even")
        return "create_msg_even"
    else: 
        print("Routing to create_msg_odd")
        return "create_msg_odd"
    

def route_based_on_4_multiple(state: GraphState) -> str:
    """Route to different nodes based on multiple of 4"""
    if state['rand_num'] % 4 ==0:
        print("Multiple of 4 received. Routing back to generate_random")
        return "generate_random"
    else: 
        print("Even number not a multiple of 4 received. Routing to end")
        return "end"


def route_based_on_prime(state: GraphState) -> str:
    """Route to different nodes based on prime"""
    if isprime(state['rand_num'] ):
        print("Prime number received. Routing back to generate_random")
        return "generate_random"
    else: 
        print("Non prime even number received. Routing to end")
        return "end"


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
workflow.add_conditional_edges(
    "create_msg_even",  # source
    route_based_on_4_multiple,
    {
        "generate_random": "generate_random",
        "end": END
    }
)
workflow.add_conditional_edges(
    "create_msg_odd",  # source
    route_based_on_prime,
    {
        "generate_random": "generate_random",
        "end": END
    }
)

#workflow.add_edge("create_msg_even", END)
#workflow.add_edge("create_msg_odd", END)

# Compile to an executable graph
workflow_compiled = workflow.compile()


# Invoke the graph
response = workflow_compiled.invoke({"max_value" : 100, "rand_num_list" : []})
print(response)