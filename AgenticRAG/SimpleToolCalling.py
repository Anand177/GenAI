from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

import os

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")

OPEN_AI_KEY=os.getenv("OPENAI_API_KEY")

tavily_tool = TavilySearch(include_raw_content=True, 
                                  include_answer=False, # Not including AI generated response
                                  include_images=False,
                                  search_depth="basic", 
                                  max_results=3)
tools = [tavily_tool]
llm=ChatOpenAI(model="gpt-4o-mini", api_key=OPEN_AI_KEY)
messages = [("user", "Who is the British PM?")]

response = llm.invoke(messages)
print(response)

agent=create_agent(model=llm, tools=tools)
response = agent.invoke({"messages": messages})
print(response)