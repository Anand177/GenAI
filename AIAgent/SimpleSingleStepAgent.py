from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI  
from langchain_tavily.tavily_search import TavilySearch

import os

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

tavily_tool = TavilySearch(include_raw_content=True, search_depth="basic", include_images=False, max_results=3,
                include_answer=False) # Not including AI generated response
                                   
llm=ChatGoogleGenerativeAI (model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY, temperature = 0.1)

prompt=ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use tools to answer user's question. Avoid using your parametric knowledge"),
    ("human", "Question: {query}\nSearch results: {search_results}\nProvide a clear answer based on the search results above.")
])

def run_single_step_agent(query: str):
    
    search_result=tavily_tool.invoke({"query" : query})
    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({"query" : query, "search_results" : search_result})
    return {"input" : query, "search_results" : search_result, "output" : response}


# Test 1 --> Test with 1 tool
query  = "Who is Anand Vasantharajan"
response = run_single_step_agent(query)
print(f"Query:: {response['input']}")
print(f"Search Result:: {response['search_results']}")
print(f"Output:: {response['output']}")
"""
# Test 2 --> When no appropriate tool is available
question = "Search web for LLM"
response = invoke_agent(question)
print(f"Question --> {question}")
print(f"Final response:: {response}")
"""