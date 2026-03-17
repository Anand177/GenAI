from dotenv import load_dotenv
from agents import Agent, ModelSettings, Runner, function_tool, trace, OutputGuardrail, GuardrailFunctionOutput

import asyncio, re

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")

@function_tool
def web_search(query: str):

    mock_search={"Mr X" : "Detective. Lives in Chennai. Usually responds to text/WhatsApp quick. Connect via 9512345678",
                 "Mr Y" : "Retired. Lives in Ooty. Not much known about the person"}
    
    return mock_search.get(query, "Record not found")

instructions = """You are a search assistant. Use available tools to search web for given term and return results. 
Return the exact results yielded by the tool. Don't enhance, summarize or truncate details received."""

search_agent = Agent(name="Search agent", instructions=instructions, tools=[web_search],
    model="gpt-4o-mini", model_settings=ModelSettings(tool_choice="required"))


async def run_agent_workflow():
    result = await Runner.run(search_agent, "Mr X")
        
    print("-" * 50)
    print("Workflow agent complete")
    print("-" * 50)
    print(f"Final Output: {result.final_output}")

"""if __name__ == "__main__":
    asyncio.run(run_agent_workflow())
"""

def no_personal_info_guardrail(context, agent, agent_output: str) -> GuardrailFunctionOutput:
    """ Blocks personal/sensitive information from the agent output.   """

    patterns = [ r"\b\d{10}\b",                  # phone numbers (basic)
                 r"\b[\w\.-]+@[\w\.-]+\.\w+\b"]  # emails

    for pattern in patterns:
        if re.search(pattern, agent_output):
            return GuardrailFunctionOutput( tripwire_triggered=True,
                 output_info="Output blocked: Contains sensitive personal information.")
    return GuardrailFunctionOutput( output_info=agent_output, tripwire_triggered=False )


search_agent_wop = Agent(name="Search agent", instructions=instructions, tools=[web_search],
    model="gpt-4o-mini", model_settings=ModelSettings(tool_choice="required"),
    output_guardrails=[OutputGuardrail(no_personal_info_guardrail)])


async def run_agent_workflow_wop():
    with trace("OP Guardrail check"):
        result = await Runner.run(search_agent_wop, "Mr X")
        
    print("-" * 50)
    print("Workflow agent complete")
    print("-" * 50)
    print(f"Final Output: {result.final_output}")

if __name__ == "__main__":
    asyncio.run(run_agent_workflow_wop())
