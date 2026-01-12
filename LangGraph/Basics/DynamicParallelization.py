from langgraph.graph import StateGraph, START, END
from typing import List, Annotated, TypedDict
from langgraph.types import Send

import operator, random

class ParentState(TypedDict):
    num_workers: int
    total: int
    numbers: Annotated[List, operator.add]      # Shared bet parent& workers

class WorkerState(TypedDict):
    worker_name: str
    range: tuple[int, int]
    numbers: Annotated[List, operator.add]      # Shared bet parent& workers


def orchestrator_node(state : ParentState):
    return {"numbers" : [], "total" : 0}        # Init Parent State

def worker_processor_node(state: WorkerState):
    print(f"Worker Name : {state['worker_name']}")
    lower, upper = state["range"]
    rint=random.randint(lower, upper)
    print(f"{state['worker_name']} Range : {lower} -> {upper}")
    print(f"{state['worker_name']} Number Generated : {rint}")
    return {"numbers" : [rint]}

# Aggregator Node
def aggregator_node(state: ParentState) -> ParentState:
    num = state["numbers"]
    return {"total" : sum(num)}

# Router Logic 
def assign_workers(state: ParentState) -> ParentState:
    num_workers = state["num_workers"]          # Get # of workers to create
    workers = []

    for i in range(num_workers):
        worker_args = {
            "worker_name" : f"Worker#{i+1}",
            "range" : (i, random.randint(1, 10+i))
        }
        workers.append(Send("worker_processor_node", worker_args))
    return workers

graph = StateGraph(ParentState)

# Add Workers
graph.add_node("orchestrator_node", orchestrator_node)
graph.add_node("worker_processor_node", worker_processor_node)
graph.add_node("aggregator_node", aggregator_node)

# Add Edges
graph.add_edge(START, "orchestrator_node")
graph.add_conditional_edges("orchestrator_node", assign_workers, ["worker_processor_node"])
graph.add_edge("worker_processor_node", "aggregator_node")
graph.add_edge("aggregator_node", END)

graph_compiled = graph.compile()
response=graph_compiled.invoke({"num_workers" : 4})

print(response)