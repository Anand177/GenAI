from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, END, StateGraph, MessagesState
from typing import List

@tool
def multiply(number_1: int, number_2: int) -> int:
    """Multiples two numbers and returns result"""
    return number_1 * number_2

@tool
def add(number_1: int, number_2: int) -> int:
    """Adds two numbers and returns result"""
    return number_1 + number_2

@tool
def subtract(number_1: int, number_2: int) -> int:
    """Subtract two numbers and returns result"""
#    number_1/0
    return number_1 - number_2

tool_node=ToolNode(tools=[multiply, add, subtract], handle_tool_errors=True)

class StateTools(MessagesState):
    number_1: int
    number_2: int
    ops: List[str]

def call_tool(state: StateTools) -> StateTools:
    tool_calls=[]
    i=1
    for op in state["ops"]:
        tool_calls.append({
            "id": str(i), "name": op, "args" : { "number_1": state["number_1"], "number_2": state["number_2"] }
        })
        i+=1
    tool_messages= AIMessage(content="Test Msg", tool_calls=tool_calls)
    messages={"messages" : [tool_messages]}

    return messages


#Create Graph
graph=StateGraph(StateTools)
graph.add_node("call_tool", call_tool)
graph.add_node("tool_node", tool_node)

graph.add_edge(START, "call_tool")
graph.add_edge("call_tool", "tool_node")
graph.add_edge("tool_node", END)

compiled_graph=graph.compile()

ops=["multiply", "add", "subtract"]
number_1=27
number_2=9

response=compiled_graph.invoke({"number_1": number_1, "number_2": number_2, "ops": ops})

print(response)