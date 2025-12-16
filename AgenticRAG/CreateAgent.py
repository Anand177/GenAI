from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

import os
import json

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

# Create LLM
llm = ChatOpenAI(model="gpt-4o-mini",
            temperature=0.3,
            top_p=0.7,
            api_key=OPENAI_API_KEY,
            frequency_penalty=1
)

@tool
def  company_stock_price(stock_symbol: str):
    """
    Retrieve the current stock price for a given company stock symbol.

    This function accepts a stock symbol and returns the current stock price for the specified company.
    It supports stock symbols for the following companies:
    - Apple Inc. ("AAPL")
    - Microsoft Corporation ("MSFT")
    - Amazon.com, Inc. ("AMZN")

    If the stock symbol does not match any of the supported companies, the function returns "unknown" as the price.

    Args:
        stock_symbol (str): The stock symbol of the company whose stock price is requested.

    Returns:
        dict: A dictionary containing the stock price with the key "price".
              If the stock symbol is not recognized, the value is "unknown".
    """
    if stock_symbol.upper()=='AAPL':
        return  {"price": 192.32}
    elif stock_symbol.upper()=='MSFT':
        return  {"price": 415.60}
    elif stock_symbol.upper()=='AMZN':
        return  {"price": 183.60}
    else:
        return {"price": "unknown"}


@tool
def city_weather(city: str) -> int:
    """
    Retrieve the current weather information for a given city.

    This function accepts a city name and returns a dictionary containing the current temperature and weather forecast for city. 
    It supports the following cities:
    - New York
    - Paris
    - London

    If the city is not recognized, the function returns "unknown" for the temperature.

    Args:
        city (str): The name of the city for which the weather information is requested.

    Returns:
        dict: A dictionary with the following keys:
            - "temperature": The current temperature in Fahrenheit.
            - "forecast": A brief description of the current weather forecast.
            
        If the city is not recognized, the "temperature" key will have a value of "unknown".
    """
    if city.lower() == "new york":
        return {"temperature": 68, "forecast": "rain"}
    elif city.lower() == "paris":
        return {"temperature": 73, "forecast": "sunny"}
    elif city.lower() == "london":
        return {"temperature": 82, "forecast": "cloudy"}
    else:
        return {"temperature": "unknown"}

# create the tools array
tools = [company_stock_price, city_weather]

# System Prompt
system = """
Respond to the human as helpfully and accurately as possible. You have access to the following tools:{tools}
Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Valid "action" values: "Final Answer" or {tool_names}
Provide only ONE action per $JSON_BLOB, as shown:

`` `
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
`` `

Follow this format:

Question: input question to answer
Thought: consider previous and subsequent steps
Action:
`` `
$JSON_BLOB
`` `
Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action:
`` `
{{
  "action": "Final Answer",
  "action_input": "Final response to human"
}}

`` `

Always respond with a valid json blob of a single action. Use tools if necessary. 
Respond directly if appropriate. Format is Action:`` `$JSON_BLOB`` `then Observation
"""

# Human Prompt
human = """
{input}
{agent_scratchpad}
(reminder to respond in a JSON blob no matter what)
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        MessagesPlaceholder("chat_history", optional=True),
        ("human" , human)
    ]
)

agent = create_structured_chat_agent(llm=llm, prompt=prompt, tools=tools)
agent_executor = AgentExecutor(agent=agent,
                               tools=tools,
                               max_execution_time=60000,
                               max_iterations=5,
                               verbose=True,
                               handle_parsing_errors=True)
question="""I am interested in investing in one of these stocks: AAPL, MSFT, or AMZN.

Decision Criteria:

Sunny Weather: Choose the stock with the lowest price.
Raining Weather: Choose the stock with the highest price.
Cloudy Weather: Do not buy any stock.
Location:

I am currently in New York.
Question:

Based on the current weather in New York and the stock prices, which stock should I invest in?
"""

response = agent_executor.invoke({"input" : question})

print(response)