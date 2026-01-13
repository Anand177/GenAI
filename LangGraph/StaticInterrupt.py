from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.types import Command
from typing import Annotated, List, TypedDict


def append_to_list(current: List[str], new: List[str]) -> List[str]:
    return current + new

class StaticInterrupt(TypedDict):
    msg: Annotated[List[str], append_to_list]

def node_a(state: StaticInterrupt) -> StaticInterrupt:
    print("In Node A")
    return {'msg': ["Hello from Node A"]}

def node_b(state: StaticInterrupt) -> StaticInterrupt:
    print("In Node B")
    return {'msg': ["Hello from Node B"]}

def node_c(state: StaticInterrupt) -> StaticInterrupt:
    print("In Node C")
    return {'msg': ["Hello from Node C"]}

statInteruptGraph=StateGraph(StaticInterrupt)

statInteruptGraph.add_node("node_a", node_a)
statInteruptGraph.add_node("node_b", node_b)
statInteruptGraph.add_node("node_c", node_c)

statInteruptGraph.add_edge(START, "node_a")
statInteruptGraph.add_edge("node_a", "node_b")
statInteruptGraph.add_edge("node_b", "node_c")
statInteruptGraph.add_edge("node_c" , END)


statInteruptGraphCompiled=statInteruptGraph.compile()
response=statInteruptGraphCompiled.invoke({"msg": ["Start Graph without Interrupts"]})
print("Running without Interrupts")
print(response)

checkpointer=InMemorySaver()

# Interrupt After
print("Interrupting After Node B")
config_aft={"configurable" : {"thread_id" : 1}}
statInteruptGraphCompiled=statInteruptGraph.compile(checkpointer=checkpointer, interrupt_after=['node_b'])
response=statInteruptGraphCompiled.invoke({"msg" : ["Starting with interrupt after B"]}, config=config_aft)
print(response)

next_node=statInteruptGraphCompiled.get_state(config=config_aft).next
print(f"Next Node to be executed -> {next_node}")

response=statInteruptGraphCompiled.invoke(None, config=config_aft)
print(response)


# Interrupt Before
print("Interrupting before Node B")
config_bef={"configurable" : {"thread_id" : 2}}
statInteruptGraphCompiled=statInteruptGraph.compile(checkpointer=checkpointer, interrupt_before=['node_b'])
response=statInteruptGraphCompiled.invoke({"msg" : ["Starting with interrupt before B"]}, config=config_bef)
print(response)

next_node=statInteruptGraphCompiled.get_state(config=config_bef).next
print(f"Next Node to be executed -> {next_node}")

response=statInteruptGraphCompiled.invoke(None, config=config_bef)  # None resumes without any update to the state
print(response)


# Resume with additional message
config_addl_msg={"configurable" : {"thread_id" : 3}}
statInteruptGraphCompiled=statInteruptGraph.compile(checkpointer=checkpointer, interrupt_before=['node_b'])
response=statInteruptGraphCompiled.invoke({"msg" : ["Adding addl Msg while resumption"]}, config=config_addl_msg)
print(response)

next_node=statInteruptGraphCompiled.get_state(config=config_addl_msg).next
print(f"Next Node to be executed -> {next_node}")

response=statInteruptGraphCompiled.invoke(
    Command(update={"msg" : ["Resuming step"]}, resume="Approved. Resuming step"), config=config_addl_msg)  
print(response)


# Skipping to another node with interrupt
config_skip={"configurable" : {"thread_id" : 4}}
statInteruptGraphCompiled=statInteruptGraph.compile(checkpointer=checkpointer, interrupt_before=['node_b'])
response=statInteruptGraphCompiled.invoke({"msg" : ["Skipping to C while resumption"]}, config=config_skip)
print(response)

next_node=statInteruptGraphCompiled.get_state(config=config_skip).next
print(f"Next Node to be executed -> {next_node}")

statInteruptGraphCompiled.update_state(config=config_skip, values=None, as_node="node_b")
response=statInteruptGraphCompiled.invoke(None, config=config_skip)  # Skipping to Node C
print(response)


