from dotenv import load_dotenv
from langchain_core.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_openai import ChatOpenAI
from langchain_tavily.tavily_search import TavilySearch

import os

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

# Create LLM
llm = ChatOpenAI(model="gpt-4o-mini",
            temperature=0.3,
            top_p=0.7,
            api_key=OPENAI_API_KEY,
            frequency_penalty=1
)


# Create Tavily Web Search Tool
tavily_tool = TavilySearch(include_raw_content=True, 
                                  include_answer=False, # Not including AI generated response
                                  include_images=False,
                                  search_depth="basic", 
                                  max_results=3)

# Create Wikipedia Tool
wikipedia_tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())


# create the tools array
tools = [tavily_tool, wikipedia_tool]

template="""
Answer the following questions as best you can. You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Do not run a tool again with same arguments, use tool responses from previous runs of the tool.
Do not use your internal knowledge

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(template) 

agent = create_react_agent(llm=llm, prompt=prompt, tools=tools)
agent_executor = AgentExecutor(agent=agent,
                               tools=tools,
                               max_execution_time=60000,
                               max_iterations=10,
                               verbose=True,
                               handle_parsing_errors=True)
#question="List Physics Nobel prize winner with a short summary on the year A. R. Rahman was born "
question="Short summary on Tamil nadu Chief minister on the year when A. R. Rahman was born "

response = agent_executor.invoke({"input" : question})

print(response)