from dotenv import load_dotenv
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI  
from langchain_tavily.tavily_search import TavilySearch

import os

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

tavily_tool = TavilySearch(include_raw_content=True, search_depth="basic", include_images=False, max_results=3,
                include_answer=False) # Not including AI generated response
wikipedia_tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
tools=[tavily_tool, wikipedia_tool]
                                   
llm=ChatGoogleGenerativeAI (model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY, temperature = 0.1)

prompt=ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use tools to answer user's question. Avoid using your parametric knowledge"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent=create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor=AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5, max_execution_time=60000)

# Test 1 --> Test with 1 tool
query  = "Who was Tamil Nadu Chief Minister during last sighting of Haley's comet"
response = agent_executor.invoke({"input" : query})
print(f"Query --> {query}")
print(f"Final response:: {response}")

"""
# Test 2 --> When no appropriate tool is available
question = "Search web for LLM"
response = invoke_agent(question)
print(f"Question --> {question}")
print(f"Final response:: {response}")
"""