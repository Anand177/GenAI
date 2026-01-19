from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import StateSnapshot
from typing import List, TypedDict, Annotated


import random, string

def append_to_list(current: List[str], new: List[str]) -> List[str]:
    return current + new

class CpState(TypedDict):
    inp: str
    inp_lst : Annotated[List[str], append_to_list]

def node_a(state: CpState) -> CpState:
    rand_str=''.join(random.choices(string.ascii_letters, k=3) )
    print(f"String -> {rand_str} generated in Node A")
    return {"inp" : rand_str}

def node_b(state: CpState) -> CpState:
    return {"inp_lst" : [state["inp"]]}

chckpt_graph = StateGraph(CpState)

chckpt_graph.add_node("node_a", node_a)
chckpt_graph.add_node("node_b", node_b)

chckpt_graph.add_edge(START, "node_a")
chckpt_graph.add_edge("node_a", "node_b")
chckpt_graph.add_edge("node_b", END)

checkpointer = InMemorySaver()
chckpt_graph_compiled=chckpt_graph.compile(checkpointer=checkpointer)

config={"configurable" : {"thread_id" : "A"}}
response=chckpt_graph_compiled.invoke({}, config=config)

print(response)


def print_snapshot(snapshot : StateSnapshot):
    print("-"*50)
    print(f"Step -> {snapshot.metadata['step']}")
    print(f"Next -> {snapshot.next}")
    print(f"Value -> {snapshot.values}")

    keys = snapshot.metadata.keys()
    for key in keys:
        print(f"\t {snapshot.metadata[key]}")

    print(f"Config -> {snapshot.config}")
    print(f"Parent Config -> {snapshot.parent_config}")
    print(f"Created at -> {snapshot.created_at}")
    print("-"*50, "\n")


def print_history(history : list[StateSnapshot]):
    print(f"History Length -> {len(history)}")
    for snapshot in history:
        print_snapshot(snapshot)


snapshot=chckpt_graph_compiled.get_state(config=config)
print(snapshot)
print_snapshot(snapshot)

# Print Full History
history=list(chckpt_graph_compiled.get_state_history(config))   # Get state History gets checkpoints
print_history(history)

# Checkout Snapshot @ Step 1
selected_state=history[1]
print_snapshot(selected_state)

# Update graph state with new value and execute
new_config=chckpt_graph_compiled.update_state(selected_state.config, values={'inp':'Test'})
result = chckpt_graph_compiled.invoke(None, config=new_config)  # Update Graph and execute from new checkpoint
print(result)

history2=list(chckpt_graph_compiled.get_state_history(new_config))
print_history(history2)