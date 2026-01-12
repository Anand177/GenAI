from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict

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
    return number_1 - number_2

tool_node=ToolNode(tools=[multiply, add, subtract])

class State(TypedDict):
    messages: Annotated[list, add_messages]


#Create Graph
graph=StateGraph(State)
graph.add_node("tools", tool_node)
graph.add_edge(START, "tools")
graph.add_edge("tools", END)

compiled_graph=graph.compile()

tool_call_msg=AIMessage(
    content="Test Run",
    tool_calls= [
        {"id":"111", "name":"multiply", "args" : {"number_1" : 6, "number_2" : 3}},
        {"id":"222", "name":"add", "args" : {"number_1" : 6, "number_2" : 3}},
        {"id":"333", "name":"subtract", "args" : {"number_1" : 6, "number_2" : 3}}
    ]
)

messages= {"messages": [tool_call_msg]}
response=compiled_graph.invoke(messages)

print(response)