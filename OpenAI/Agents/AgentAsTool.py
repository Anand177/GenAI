import asyncio, os, sys

sys.path.append(os.path.abspath("c:/Learning/Python/GenAI"))

from AgenticRAG.GMail.GmailTool import send_mail

from agents import Agent, Runner, trace, function_tool
from dotenv import load_dotenv
from openai.types.responses import ResponseTextDeltaEvent
from typing import Dict

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

#send_mail(content="This is a test email from Agent2.py", subject="Test Email")
# FUnction as tool
@function_tool
def send_email(content: str, subject: str = "Test EMail") -> str:
    """Send email with given content and subject"""
    send_mail(content=content, subject=subject)
    return {"status" : "success"}


instructions1 = """You are a sales agent working for ComplAI, 
a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits, powered by AI. 
You write professional, serious cold emails."""

instructions2 = """You are a humorous, engaging sales agent working for ComplAI, 
a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits, powered by AI. 
You write witty, engaging cold emails that are likely to get a response."""

instructions3 = """You are a busy sales agent working for ComplAI, 
a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits, powered by AI. 
You write concise, to the point cold emails."""

sales_agent_1 = Agent(name="Professional Sales Agent", model="gpt-4o-mini",
        instructions=instructions1)

sales_agent_2 = Agent(name="Engaging Sales Agent", model="gpt-4o-mini",
        instructions=instructions2)

sales_agent_3 = Agent(name="Busy Sales Agent", model="gpt-4o-mini",
        instructions=instructions3)

sales_picker = Agent(
    name="sales_picker",
    instructions="You pick the best cold sales email from the given options. \
Imagine you are a customer and pick the one you are most likely to respond to. \
Do not give an explanation; reply with the selected email only.",
    model="gpt-4o-mini"
)

message="Write a cold sales email"

async def agent_1_fn():
    result= Runner.run_streamed(sales_agent_1, input=message)
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)
    print()

    with trace("Parallel emails"):
        results = await asyncio.gather(
            Runner.run(sales_agent_1, message),
            Runner.run(sales_agent_2, message),
            Runner.run(sales_agent_3, message)
        )

    outputs= [result.final_output for result in results]
    for output in outputs:
        print(output + "\n\n")
    output_email=f"""Cold Sales Emails:{"\n".join(outputs)}"""
    best = await Runner.run(sales_picker, output_email)
    print(f"Final Output: {best.final_output}")

#asyncio.run(agent_1_fn())


# Using agents as tool
description="Write a cold sales email"
tool1=sales_agent_1.as_tool(tool_name="Sales_Agent1", tool_description=description)
tool2=sales_agent_2.as_tool(tool_name="Sales_Agent2", tool_description=description)
tool3=sales_agent_3.as_tool(tool_name="Sales_Agent3", tool_description=description)

tools=[tool1, tool2, tool3, send_email]
print(tools)

instructions = """
You are a Sales Manager at ComplAI. Your goal is to find the single best cold sales email using the sales_agent tools.
 
Follow these steps carefully:
1. Generate Drafts: Use all three sales_agent tools to generate three different email drafts. Do not proceed until all three drafts are ready.
2. Evaluate and Select: Review the drafts and choose the single best email using your judgment of which one is most effective.
3. Use the send_email tool to send the best email (and only the best email) to the user.
 
Crucial Rules:
- You must use the sales agent tools to generate the drafts — do not write them yourself.
- You must send ONE email using the send_email tool — never more than one.
"""

sales_manager=Agent(name="Sales_Manager", instructions=instructions, tools=tools, model="gpt-4o-mini")
message="Send a cold sales email addressed to 'Dear CEO'"

async def sm_agent():
    with trace("Sales manager"):
        result=await Runner.run(sales_manager, message)

asyncio.run(sm_agent())