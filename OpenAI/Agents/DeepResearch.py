from agents import Agent, Runner, WebSearchTool, trace, function_tool
from agents.model_settings import ModelSettings
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Dict

import asyncio, os

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

"""OpenAI WebSearchTool costs $0.03 per query. Tavily is free"""

search_instructions = """You are a research assistant. Given a search term, you search the web for that term and 
produce a concise summary of the results. The summary must have 2-3 paragraphs and less than 300 
words. Capture the main points. Write succintly, no need to have complete sentences or good 
grammar. This will be consumed by someone synthesizing a report, so it's vital you capture the 
essence and ignore any fluff. Do not include any additional commentary other than the summary itself."""

search_agent=Agent(name= "Search Agent", instructions=search_instructions, model="gpt-4o-mini",
                   model_settings=ModelSettings(tool_choice="required"),    # Force Agent to use tool rather parametric knowledge
                   tools=[WebSearchTool(search_context_size="low")]) # low is cheapest plan
                   
message="Latest AI frameworks in 2026"
async def run_web_tool():
    with trace("Running Open AI Web Tool"):
        result=await Runner.run(search_agent, message)
        print(result.final_output)

#asyncio.run(run_web_tool())

search_num=3
instructions2=f"You are a helpful research assistant. Given a query, come up with a set of web searches \
to perform to best answer the query. Output {search_num} terms to query for."

class WebSearchItem(BaseModel):
    query: str = Field(description="The search term to use for web search.")
    reason: str = Field(description="Reasoning for why this search is important to the query.")
    
class WebSearchList(BaseModel):
    searches: list[WebSearchItem] = Field(description="A list of web searches to perform to best answer the query.")


planner_agent = Agent(name="PlannerAgent", instructions=instructions2, model="gpt-4o-mini", output_type=WebSearchList,
                      tools=[WebSearchTool(search_context_size="low")], 
                      model_settings=ModelSettings(tool_choice="required"))
async def run_web_tool2():
    with trace("Search"):
        result = await Runner.run(planner_agent, message)
        print(result.final_output)

#asyncio.run(run_web_tool2())

writer_instructions = """You are a senior researcher tasked with writing a cohesive report for a research query.
    You will be provided with the original query, and some initial research done by a research assistant.
    You should first come up with an outline for the report that describes the structure and 
    flow of the report. Then, generate the report and return that as your final output.
    The final output should be in markdown format, and it should be lengthy and detailed. Aim 
    for 5-10 pages of content, at least 1000 words."""

class ReportData(BaseModel):
    short_summary: str = Field(description="A short 2-3 sentance summary of findings")
    markdown_report: str = Field(description="The final report")
    follow_up_questions: str = Field(description="Suggested topics to research further")

writer_agent=Agent(name="WriterAgent", instructions=writer_instructions, model="gpt-4o-mini", output_type=ReportData)


@function_tool
def send_email(subject: str, html_body: str) -> Dict[str, str]:
    """ Mock function to print mail """
    print(f"Subject -> {subject}")
    print(f"Html Body:\n{html_body}")
    return {"status" : "success" }

email_instructions = """You are able to send a nicely formatted HTML email based on a detailed report.
You will be provided with a detailed report. You should use your tool to send one email, providing the 
report converted into clean, well presented HTML with an appropriate subject line."""

email_agent = Agent(name="Email agent", instructions=email_instructions, tools=[send_email], model="gpt-4o-mini")



async def run_planner_agent(query: str):
    """using Planner Agent to plan which searches to run for query"""
    print("Planning Searches")
    result = await Runner.run(planner_agent, f"Query: {query}")
    print(f"Will perform {len(result.final_output.searches)} searches")
    return result.final_output


async def search(item: WebSearchItem):
    """Invoke run_search_agent() for each item in search_plan"""
    input = f"""Search item: {item.query}
                Search Reason: {item.reason}"""
    result= await Runner.run(search_agent, input)

async def perform_searches(search_plan : WebSearchList):
    """Invoke search() for each item in search plan"""
    print("Searching")
    num_completed=0
    tasks = [asyncio.create_task(search(item)) for item in search_plan.searches]
    results = await asyncio.gather(*tasks)
    print("Finished Searching")
    return results

async def write_report(query: str, search_results: list[str]):
    """Use writer agent to write report based on search results"""
    print("Writing report")
    input =f"""Original Query: {query}
            Summarized search results: {search_results}"""
    result = await Runner.run(writer_agent, input)
    print("Finished writing report")
    return result.final_output

async def run_email_agent(report: ReportData):
    """ Use the email agent to send an email with the report """
    print("Writing email...")
    result = await Runner.run(email_agent, report.markdown_report)
    print("Email sent")
    return report

query="Latest AI agent frameworks in 2026"
async def run_agents():
    with trace("Research trace:"):
        print("starting research")
        search_plan = await run_planner_agent(query=query)
        search_results = await perform_searches(search_plan=search_plan)
        report = await write_report(query=query, search_results = search_results)
        await run_email_agent(report)
        print("Complete!")

asyncio.run(run_agents())