from dotenv import load_dotenv
from agents import Agent, ModelSettings, Runner, function_tool, trace, OutputGuardrail, GuardrailFunctionOutput, OutputGuardrailResult

import asyncio, re

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")

@function_tool
def web_search(query: str):

    mock_search={"Mr X" : "40 years old. Lives in Chennai. Usually responds to text/WhatsApp quick. Connect via 9512345678",
                 "Mr Y" : "Retired. Lives in Ooty. Not much known about the person",
                 "Elephant" : "Largest land mammal. Lives in Asia/ Africa"}
    
    return mock_search.get(query, "Record not found")

guardrail_instructions = """You are a safety guardrail. 
Check if the output contains personal or sensitive information about a person.

Disallowed content:
- Phone numbers, emails, addresses
- Exact age, DOB

Allowed:
- Public, high-level info about well-known public figures (no private details)

If SAFE:
Return the original text unchanged.

If UNSAFE:
Return: "Output blocked: Contains personal or sensitive information."
"""

guardrail_agent = Agent(name="Output Guardrail", instructions=guardrail_instructions, model="gpt-4o-mini",
    model_settings=ModelSettings(temperature=0))

async def llm_guardrail_fn(context, agent, agent_output) -> GuardrailFunctionOutput:
    review = await Runner.run(guardrail_agent, f"Review this output:\n\n{agent_output}" )
    reviewed_text = review.final_output.strip()

    if reviewed_text.startswith("Output blocked"):
        return GuardrailFunctionOutput( tripwire_triggered=True, output_info=reviewed_text)

    return GuardrailFunctionOutput(tripwire_triggered=False, output_info=agent_output)

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
    print(f"Final Output: {result.final_output}")

if __name__ == "__main__":
    asyncio.run(run_agent_workflow_wop())
