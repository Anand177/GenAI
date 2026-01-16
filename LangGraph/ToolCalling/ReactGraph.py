#Refer ReactGraph.ipynb
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from pydantic import BaseModel

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

#Create Prompt
prompt="You are a helpful middle school math teacher. Answer basic math questions only. " \
"Decline to answer on other topics"

# Create structured Output 
class MathResult(BaseModel):
    answer: int
    explanation: str

chat_model = init_chat_model("gpt-4o-mini")
agent = create_agent(model=chat_model, tools=[add, subtract, multiply],
            system_prompt=prompt, response_format=MathResult, debug=False)

#Invoke agent
def invoke_agent(agent, human_message):
    input_state={"messages" : [HumanMessage(content=human_message)]}
    response = agent.invoke(input_state)
    return response

response=invoke_agent(agent, "200*200")
print(response)
print(response["messages"][-1].content)

response=invoke_agent(agent, "When were you last updated")
#print(response)
print(response["messages"][-1].content)