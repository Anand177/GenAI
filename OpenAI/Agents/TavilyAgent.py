from dotenv import load_dotenv
from agents import Agent, ModelSettings, Runner, function_tool
from langchain_tavily.tavily_search import TavilySearch

import asyncio

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")

tavily_client = TavilySearch(include_raw_content=True, search_depth="basic", max_results=3, include_images=False,
                    include_answer=False) # Not including AI generated response

@function_tool
def tavily_search(query: str):
    results=tavily_client.run(query)
    return results

instructions = """You are a research assistant. Given a search term, search the web for that term and 
    produce a concise summary of the results. The summary must be 2-3 paragraphs and less than 300 words.
    Capture the main points. Write succintly, no need to have complete sentences or good grammar. 
    This will be consumed by someone synthesizing a report, so its vital you capture the 
    essence and ignore any fluff. Do not include any additional commentary other than the summary itself."""

search_agent = Agent(name="Search agent", instructions=instructions, tools=[tavily_search],
    model="gpt-4o-mini", model_settings=ModelSettings(tool_choice="required"))


async def run_agent_workflow():
    result = await Runner.run(search_agent, "Who is Kendra Hillert")
        
    print("-" * 50)
    print("Workflow agent complete")
    print("-" * 50)
    print(f"Result: {result}")
    print(f"Final Output: {result.final_output}")
    print(f"Final Output: {result.raw_responses}")

if __name__ == "__main__":
    asyncio.run(run_agent_workflow())

