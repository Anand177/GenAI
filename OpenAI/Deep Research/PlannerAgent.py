from dotenv import load_dotenv
from agents import Agent
from pydantic import BaseModel, Field

import os

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

num_searches=3
instructions = f"""You are a helpful research assistant. 
Derive set of web searches for given query to get best answer for the query. 
Output {num_searches} terms to query for."""

class WebSearchItem(BaseModel):
    reason: str = Field(description="Reasoning for why this search is important to the query.")
    query: str = Field(description="The search term to use for the web search.")


class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem] = Field(description="List of web searches to perform to best answer the query.")
    
planner_agent = Agent(name="PlannerAgent", instructions=instructions, model="gpt-4o-mini", output_type=WebSearchPlan)