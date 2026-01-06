# Using Reducer function for Graph
from langgraph.graph import StateGraph, START, END
from typing import Annotated, List, TypedDict

import operator

# Reducer function to append message to List
def append_to_list(current: List[str], new: List[str]) -> List[str]:
    return current + new

class StateAnnotated(TypedDict):
    messages: Annotated[List[str], append_to_list]
    total_letters: Annotated[int, operator.add]


def add_message_annotated(state: StateAnnotated) -> StateAnnotated:
    message = f"Dummy Message. Total letters = {state['total_letters']}"
    return {"messages" : [message]}

def inc_letter_count_annotated(state: StateAnnotated) -> StateAnnotated:
    total_letter = len(state['messages'][-1])
    return {"total_letters" : total_letter}

workflow_state_annotated = StateGraph(StateAnnotated)

# Add nodes
workflow_state_annotated.add_node("add_message_annotated", add_message_annotated)
workflow_state_annotated.add_node("inc_letter_count_annotated", inc_letter_count_annotated)

# Add edges
workflow_state_annotated.add_edge(START, "add_message_annotated")
workflow_state_annotated.add_edge("add_message_annotated", "inc_letter_count_annotated")
workflow_state_annotated.add_edge("inc_letter_count_annotated", END)

workflow_state_compiled = workflow_state_annotated.compile()

response = workflow_state_compiled.invoke({"messages" : [], "total_letters" : 0})
print(response)

response = workflow_state_compiled.invoke(response)
print(response)