# Refer SimpleGraphLoop.ipynb for visualization
from langgraph.graph import START, END, StateGraph
from typing import TypedDict

# State Schema to implement loop
class BasicLoop(TypedDict):
    itr_num : int
    curr_itr : int

# Router logic to decide loop continuation/termination
def route_logic(state: BasicLoop) -> str:
    if state['curr_itr'] < state['itr_num']:
        return "continue"
    else:
        return "exit"
    
# Dummy decision node
def decision_node(state: BasicLoop) -> dict:
    return {}

# process for iteration
def processor (state: BasicLoop) -> dict:
    itr = state['curr_itr'] +1
    print(f"In Iteration -> {itr}")
    return {"curr_itr" : itr}


wf_loop = StateGraph(BasicLoop)

# Adding nodes
wf_loop.add_node("decision_node", decision_node)
wf_loop.add_node("processor", processor)

wf_loop.add_edge(START, "decision_node")

wf_loop.add_conditional_edges("decision_node", route_logic,
                              {"continue" : "processor",
                               "exit" : END})
wf_loop.add_edge("processor", "decision_node")


wf_loop_compiled = wf_loop.compile()

response = wf_loop_compiled.invoke({"itr_num": 6, "curr_itr": 0})
print(response)