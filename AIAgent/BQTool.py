from dotenv import load_dotenv

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import BaseTool
from langchain_google_genai import GoogleGenerativeAI
from sqlalchemy import create_engine
from typing import List

import os

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")


os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="C:/Learning/AI/Key/gen-lang-client-0398817262-d752d424f736.json"
project_id="gen-lang-client-0398817262"
dataset_id="GenAi"
table_name="test_table"
query=f"SELECT * FROM `{project_id}.{dataset_id}.{table_name}` WHERE id = 123"

bq_uri=f"bigquery://{project_id}/{dataset_id}"


def list_bq_tools(bq_toolkit : SQLDatabaseToolkit):
    
    tools = bq_toolkit.get_tools()
    print("-"*50)
    print("BQ Tools")
    print("-"*50)

    for i, tool in enumerate(tools):
        print(f"Tool #{i}")
        print(f"Tool Name: {tool.name}")
        print(f"Tool Desc: {tool.description}")
        if hasattr(tool, "func"):
            print(f"Function: {tool.func.__name__}")
        print("-"*50)

    print(f"Total tools :{len(tools)}")
    return tools


def run_tool(toolkit : SQLDatabaseToolkit, tool_name: str, tool_input):
    tools : List[BaseTool] = toolkit.get_tools()
    target_tool = next(tool for tool in tools if tool.name == tool_name)
    result = target_tool.run(tool_input)

    print("-"*50)
    print(f"Tool Name -> {tool_name}")
    print(f"Input -> {tool_input}")
    print(f"Output -> {result}")
    print("-"*50)


def list_tables(bq_toolkit : SQLDatabaseToolkit):
    tool_name="sql_db_list_tables"
    run_tool(bq_toolkit,tool_name,{})

def get_table_schema(bq_toolkit : SQLDatabaseToolkit):
    tool_name="sql_db_schema"
    run_tool(bq_toolkit,tool_name,table_name)

def run_query(bq_toolkit : SQLDatabaseToolkit):
    tool_name="sql_db_query"
    run_tool(bq_toolkit,tool_name,query)


def run_agent_query(bq_toolkit : SQLDatabaseToolkit, query : str):
    """Convert Natural language to BQ query and execute"""
    agent_executor= create_sql_agent(llm=bq_toolkit.llm,
                        toolkit=bq_toolkit,
                        verbose=True,           # Show agent's thought process
                        handle_parsing_errors=True)
    print("-"*50)
    print("Converting natural language Query to BQ query")
    print("-"*50)

    result=agent_executor.invoke({"input": query})
    print(result.get('output'))


try:
    engine=create_engine(bq_uri)
    db=SQLDatabase(engine)

    llm=GoogleGenerativeAI(model="gemini-flash-latest", google_api_key=GOOGLE_API_KEY)

    bq_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    bq_tools=list_bq_tools(bq_toolkit)

    list_tables(bq_toolkit)
    get_table_schema(bq_toolkit)
    run_query(bq_toolkit)

    natural_language_query="Get details of Anand from test_table"
    run_agent_query(bq_toolkit, natural_language_query)


except Exception as e:
    print(e)