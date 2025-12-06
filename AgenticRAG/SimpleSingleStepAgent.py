from dotenv import load_dotenv
from langchain_classic.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI 

import json
import os


load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
llm=GoogleGenerativeAI(model="gemini-flash-latest",
            google_api_key=GOOGLE_API_KEY,
            temperature = 0.7
)


# Tool 1 for stocks
def company_stock_price(stock_symbol: str) -> float:
    stock_dict = {
        "AAPL" : 192.32,
        "MSFT" : 415.02,
        "AMZN" : 136.70
    }
    if(stock_symbol.upper in stock_dict):
        return {"price" : stock_dict[stock_symbol]}
    else:
        return {"price" : "unknown"}

stock_tool_description = {
    "name" : "company_stock_price",
    "description": "This tool returns the last known stock price for a company based on its ticker symbol",
    "arguments": [
        {"stock_symbol" : "stock ticker symbol for the company"}
    ],
    "response": "last known stock price"
}


## Tool 2 for city weather
def city_weather(city: str) -> int:

    weather_dict = {
        "new york": {"temperature": 68, "forecast": "rain"},
        "paris" : {"temperature": 73, "forecast": "sunny"},
        "london" : {"temperature": 82, "forecast": "cloudy"},

    }
    if city.lower() in weather_dict:
        return weather_dict[city.lower()]
    else:
        return {"temperature": "unknown"}
        
city_weather_tool_description = {
    "name" : "city_weather",
    "description": "This tool returns the current temperature and forecast for the given city",
    "arguments": [
        {"city" : "name of the city"}
    ],
    "response": "current temperature & forecast"
}

# Maintain the tools in a map for invocation by th eagent
tools = [stock_tool_description, city_weather_tool_description]
tools_map = {
    'company_stock_price': company_stock_price,
    'city_weather' : city_weather
}


template = """
You are a helpful assistant capable of answering questions on various topics. 
Do not use your internal knowledge or information to answer questions.

Instructions:

Think step-by-step to create a plan.
Use only the following available tools to find information.
Tools Available: {tools}

Guidelines for Responses:

Format_1: If the question cannot be answered with the available tools, use this format:
{{"answer": "No appropriate tool available"}}

Format_2: If you need to run tools to obtain the information, use this format:
{{"actions": [{{ "action" : tool name, "arguments" : dictionary of argument values}}]}}

Format_3: If you can answer the question using the responses from the tools, use this format:
{{"answer": "your response to the question", "explanation": "provide your explanation here"}}

Avoid any preamble; respond directly using one of the specified JSON formats.
Question: {question}

Tool Responses:{tool_responses}

Your Response:
"""

prompt=PromptTemplate(
    template=template,
    input_variables= ['tools', 'question', 'tool_responses']
)

# Create Agent RAG
def invoke_agent(question):

    # Step 1 --> Invoke LLM to generate Plan
    query=prompt.format(tools=tools, question=question, tool_responses="")
    response = llm.invoke(query)

    response_json=json.loads(response) # Convert response from AIMessage to JSON fmt
    print("Step 1 LLM Plan")
    print(f" --> {response_json}")

    # Step 2 --> Invoke tools 
    action_responses=[]
    if "answer" in response_json:
        return {"answer" : response_json["answer"]}
    elif "actions" in response_json:
        action_responses=invoke_tools(response_json)

    print("Step 2 Agent Tool invocation responses")
    print(f" --> {action_responses}")

    # Step 3 --> Invoke LLM for final response
    query=prompt.format(tools=tools, question=question, tool_responses=action_responses)
    response= llm.invoke(query)
    response_json=json.loads(response) # Convert response from AIMessage to JSON fmt
    print("Step 3 Final LLM Response")
    print(f" --> {response_json}")

    # Extract answer from response
    if "answer" in response_json:
        return response_json["answer"]
    else:
        return ("Can't generate as there is no response from the tool!!!")
    

def invoke_tools(response):
    action_responses = []
    if(len(response["actions"]) ==0):
        print('question cannot be answered as there is no tool to use !!!')
        exit
    else:
        for action in response["actions"]:
            # Get function pointer from tool map
            action_function= tools_map[action["action"]]

            # Invoke tool/with argument suggested by LLM
            action_invoke_result=action_function(**action["arguments"])
            action["response"]=action_invoke_result

            action_responses.append(action)

    return action_responses


# Test 1 --> Test with 1 tool
question = "Which of these cities is hotter, Paris or London"
response = invoke_agent(question)
print(f"Question --> {question}")
print(f"Final response:: {response}")


# Test 2 --> When no appropriate tool is available
question = "Search web for LLM"
response = invoke_agent(question)
print(f"Question --> {question}")
print(f"Final response:: {response}")


# Test 3 --> When more than 1 tool needed
question = """
I am interested in investing in one of these stocks: AAPL, MSFT, or AMZN.

Decision Criteria:
Sunny Weather: Choose the stock with the lowest price.
Raining Weather: Choose the stock with the highest price.
Cloudy Weather: Do not buy any stock.
Location:

I am currently in New York.
Question: Based on the current weather in New York and the stock prices, which stock should I invest in?
"""
response = invoke_agent(question)
print(f"Question --> {question}")
print(f"Final response:: {response}")

 
# Test 4 --> When tool response is insufficient
question = "I have $200 can i buy GOOGL stock"
response = invoke_agent(question)
print(f"Question --> {question}")
print(f"Final response:: {response}")
