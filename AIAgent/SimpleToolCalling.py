from dotenv import load_dotenv

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

import os

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")

OPEN_AI_KEY=os.getenv("OPENAI_API_KEY")



llm=ChatOpenAI(model="gpt-4o-mini", api_key=OPEN_AI_KEY)
messages = [("user", "Who is the British PM?")]

tavily_tool = TavilySearch(include_raw_content=True, 
                                  include_answer=False, # Not including AI generated response
                                  include_images=False,
                                  search_depth="basic", 
                                  max_results=3)
tools = [tavily_tool]
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use your tools when necessary to answer questions accurately."),
    MessagesPlaceholder(variable_name="messages"),  
    MessagesPlaceholder(variable_name="agent_scratchpad"), 
])


response = llm.invoke(messages)
print(response)

agent=create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

response = agent_executor.invoke({"messages": messages})
print(response)