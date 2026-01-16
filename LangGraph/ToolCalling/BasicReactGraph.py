#Refer BasicReactGraph.ipynb
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, END, MessagesState, StateGraph

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")

# Create Tools
@tool
def multiply(number_1: int, number_2: int) -> int:
    """Multiple Two Integers"""
    return number_1 * number_2

@tool
def add(number_1: int, number_2: int) -> int:
    """Add Two Integers"""
    return number_1 + number_2

@tool
def subtract(number_1: int, number_2: int) -> int:
    """Subtract Two Integers"""
    return number_1 - number_2

tool_node=ToolNode(name="math_tools",
            tools=[multiply, add, subtract])

chat_model = init_chat_model("gpt-4o-mini")
llm_with_tools=chat_model.bind_tools([add, subtract, multiply])
print(llm_with_tools)

"""
input_state = {"messages": [HumanMessage(content="what is 3*2 and 5+2")]}
response=llm_with_tools.invoke(input_state["messages"])
print(response)
"""

# Create Agent Node
def  agent_node(state : MessagesState):
     messages = state["messages"]
     response = llm_with_tools.invoke(messages)
     return {"messages": response}

def call_router(state: MessagesState):
    
    last_msg = state["messages"][-1]
    # Route to Tool Node if last message is AIMessage & there is AI request to invoke tools
    if type(last_msg) == AIMessage and len(last_msg.tool_calls) > 0:
        print("Routing to Tool Node")
        return "tool_node"
    else:
        print("Routing to End")
        return "End"

# Setup Graph
graph=StateGraph(MessagesState)

graph.add_node("agent_node", agent_node)
graph.add_node("tool_node", tool_node)

graph.add_edge(START, "agent_node")
graph.add_conditional_edges("agent_node", call_router, {"End":END, "tool_node": "tool_node"})
graph.add_edge("tool_node", "agent_node")

graph_compiled=graph.compile()

input_state = {"messages": [HumanMessage(content="Calculate 5 * 3 and 20+65 then subtract the result of 5*3 from the other result")]}
response = graph_compiled.invoke(input_state)
print(response)
print(response["messages"][-1].content)
