from dotenv import load_dotenv
from agents import Agent, ModelSettings, Runner, function_tool, trace, OutputGuardrail, GuardrailFunctionOutput

import asyncio, re

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")

@function_tool
def web_search(query: str):

    mock_search={"Mr X" : "40 years old. Lives in Chennai. Usually responds to text/WhatsApp quick. Connect via 9512345678",
                 "Mr Y" : "Retired. Lives in Ooty. Not much known about the person"}
    
    return mock_search.get(query, "Record not found")

guardrail_sanitize_instructions = """You are a safety guardrail. If the text contains personal/sensitive information:
- Remove or generalize it (e.g., remove email, replace exact age with 'adult', replace phone number with XXXXXXXXXX)
- Keep the rest of the summary intact

If safe:
Return unchanged.

Return ONLY the cleaned text.
"""

guardrail_agent = Agent(name="Output Guardrail", instructions=guardrail_sanitize_instructions, model="gpt-4o-mini",
    model_settings=ModelSettings(temperature=0))

async def llm_guardrail_fn(context, agent, agent_output) -> GuardrailFunctionOutput:
    review = await Runner.run(guardrail_agent, f"Review this output:\n\n{agent_output}" )
    print("RAW OUTPUT:", agent_output)
    print("SANITIZED:", review.final_output)

    return GuardrailFunctionOutput( output_info=review.final_output, tripwire_triggered=False )


instructions = """You are a search assistant. Use tools and return guardrail's output.
Do not add formatting, bullets, headings, or explanations."""

search_agent_wop = Agent(name="Search agent", instructions=instructions, tools=[web_search],
    model="gpt-4o-mini", model_settings=ModelSettings(tool_choice="required"),
    output_guardrails=[OutputGuardrail(llm_guardrail_fn)])


async def run_agent_workflow_wop():
    with trace("OP Guardrail check"):
        result = await Runner.run(search_agent_wop, "Mr Y")

        
    print("-" * 50)
    print("Workflow agent complete")
    print("-" * 50)
# As trip wire isnt triggered, output is manually fetched 
    if result.output_guardrail_results and len(result.output_guardrail_results) > 0:
        sanitized_output = result.output_guardrail_results[0].output
        print(f"Final Output: {sanitized_output}")
    else:
        print(f"Final Output: {result.final_output}")

if __name__ == "__main__":
    asyncio.run(run_agent_workflow_wop())
