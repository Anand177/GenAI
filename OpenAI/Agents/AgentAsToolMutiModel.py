from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, Runner, trace, function_tool, OpenAIChatCompletionsModel, input_guardrail, GuardrailFunctionOutput
from typing import Dict
from pydantic import BaseModel

import os, asyncio

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY=os.getenv("GOOGLE_API_KEY")

instructions1 = "You are a sales agent working for ComplAI, \
a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits, powered by AI. \
You write professional, serious cold emails."

instructions2 = "You are a busy sales agent working for ComplAI, \
a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits, powered by AI. \
You write concise, to the point cold emails."

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

gemini_client=AsyncOpenAI(base_url=GEMINI_BASE_URL, api_key=GOOGLE_API_KEY)
groq_client=AsyncOpenAI(base_url=GROQ_BASE_URL, api_key=GROQ_API_KEY)

gemini_model=OpenAIChatCompletionsModel(model="gemini-2.5-flash", openai_client=gemini_client)
groq_model=OpenAIChatCompletionsModel(model="llama-3.3-70b-versatile", openai_client=groq_client)

sales_agent1=Agent(name="Gemini Sales Agent", instructions=instructions1, model=gemini_model)
sales_agent2=Agent(name="Groq Sales Agent", instructions=instructions2, model=groq_model)

description="Write an AI sales email"

tool1=sales_agent1.as_tool(tool_name="sales_agent1", tool_description=description)
tool2=sales_agent2.as_tool(tool_name="sales_agent2", tool_description=description)

@function_tool
def send_email(subject: str, html_body: str) -> Dict[str, str]:
    """This tool mocks send email functionality"""
    print("*** In send_html_email function ***")
    print(f"Subject : {subject}")
    print(f"Body:\n{html_body}")
    return {"status": "success"}

tools=[tool1, tool2, send_email]

instructions = """
You are a Sales Manager at ComplAI. Your goal is to find the single best cold sales email using the sales_agent tools.
 
Follow these steps carefully:
1. Generate Drafts: Use two sales_agent tools to generate three different email drafts. Do not proceed until all two drafts are ready.
2. Evaluate and Select: Review the drafts and choose the single best email using your judgment of which one is most effective.
3. Use the send_email tool to send the best email (and only the best email) to the user.
 
Crucial Rules:
- You must use the sales agent tools to generate the drafts — do not write them yourself.
- You must send ONE email using the send_email tool — never more than one.
"""

sales_manager=Agent(name="Sales_Manager", instructions=instructions, tools=tools, model="gpt-4o-mini")
message="Send a cold sales email addressed to 'Dear Michael' from Anand V"
"""
async def sm_agent():
    with trace("Sales manager"):
        result=await Runner.run(sales_manager, message)
        print(result.final_output)

asyncio.run(sm_agent())
"""

##Using Guardrail
class NameCheckOutput(BaseModel):
    is_name_in_msg: bool
    name: str

guardrail_instruction="Check if user is including somone's personal name in what they want to do"
guardrail_agent=Agent(name="Name Check", output_type=NameCheckOutput, model="gpt-4o-mini", instructions=guardrail_instruction)

@input_guardrail
async def guardrail_against_name(ctx, agent, message):
    result = await Runner.run(guardrail_agent, message, context=ctx.context)
    is_name_in_message = result.final_output.is_name_in_msg
    return GuardrailFunctionOutput(output_info={"found_name": result.final_output}, tripwire_triggered=is_name_in_message)

careful_sales_manager=Agent(name="Careful Sales manager", instructions=instructions, tools=tools, 
                            model="gpt-4o-mini", input_guardrails=[guardrail_against_name])

async def careful_sm_agent():
    with trace("Careful Sales manager"):
        result=await Runner.run(careful_sales_manager, message)
        print(result.final_output)

asyncio.run(careful_sm_agent())
