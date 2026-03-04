from agents import Agent, Runner, trace, function_tool
from dotenv import load_dotenv
from openai.types.responses import ResponseTextDeltaEvent
from typing import Dict

import asyncio, os

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

@function_tool
def send_html_email(subject: str, html_body: str) -> Dict[str, str]:
    """This tool mocks send email functionality"""
    print("*** In send_html_email function ***")
    print(f"Subject : {subject}")
    print(f"Body:\n{html_body}")
    return {"status": "success"}

subject_instructions = """You are an expert at writing compelling email subject lines.
When you receive email content, analyze it and create a short, attention-grabbing subject line 
that will make an executive want to open the email
CRITICAL: Return ONLY the subject line text - no explanations, no "Subject:", just the text itself.
Example good response: "Transform Your Security with AI in 30 Days"
Example bad response: "Here's a subject line: Transform Your Security..." """

html_instructions = """You are an expert at converting plain text emails to beautiful HTML.

When you receive plain text email content, convert it to well-formatted HTML with:
- Professional styling
- Proper paragraph spacing
- Clear structure
- Simple, clean design

CRITICAL: Return ONLY the HTML code - no explanations, no markdown code fences, just pure HTML.
Start directly with HTML tags like <html> or <div>"""

subject_writer = Agent(name="Email Subject Writer", model="gpt-4o-mini", instructions=subject_instructions)
html_converter = Agent(name="HTML Email Body converter", model="gpt-4o-mini", instructions=html_instructions)

subject_tool=subject_writer.as_tool(tool_name="subject_writer", tool_description=
        """REQUIRED FIRST STEP: Generates email subject line. Input: the email body text. Output: subject line only."""
)
html_tool=html_converter.as_tool(tool_name="html_converter", tool_description=
        """REQUIRED SECOND STEP: Converts plain text to HTML. 
        Input: the email body text. 
        Output: HTML formatted email.""")
tools=[subject_tool, html_tool, send_html_email]

email_instructions="""You are an email formatter and sender. When you receive an email body from the Sales Manager:
1. First, you must use subject_writer tool to generate a subject line. Pass the email body text as the input parameter
        - Wait for and save the subject line result
2. Then, you must use the html_converter tool to convert the body to HTML (pass the email body as input)
3. Finally, use send_html_email tool to send the email with the subject and HTML body.
        - Use Subject from Step 1
        - Use HTML Body from Step 2

DO NOT skip any steps. DO NOT generate subject or HTML yourself. 
You MUST use the tools in this exact order: subject_writer → html_converter → send_html_email"""

email_agent = Agent(name="Email Manager", instructions=email_instructions, tools=tools, model="gpt-4o-mini",
                    handoff_description="Receives email content, formats it as HTML, and sends it")
handoffs=[email_agent]

sales_manager_instruction="""You are a sales manager at AAR AI, an innovative AI security solutions company.

WORKFLOW:
1. First, compose a compelling cold sales email with these requirements:
   - Address it to "Dear Mr. CEO"
   - Highlight AAR AI's AI security capabilities and value proposition
   - Keep it concise and professional (3-4 paragraphs)
   - Sign it "Best regards, Anand"
   
2. After you've composed the email, you MUST hand off to the Email Manager agent
   - When handing off, include the complete email body text in your message
   - Make it clear that this is the email content to be formatted and sent

DO NOT just say "send an email" - actually write out the full email text first."""
sales_manager = Agent(name="Sales Manager", instructions=sales_manager_instruction, handoffs=handoffs, model="gpt-4o-mini")

async def run_agent_workflow():
    message = "Send out a cold sales email addressed to Mr CEO from Anand"
    print("Starting agent workflow...")
    print("-" * 50)
    with trace("Sales Email Handoff"):
        result = await Runner.run(sales_manager, message)
        
    print("-" * 50)
    print("Workflow agent complete")
    print("-" * 50)
    print(f"Final Output: {result.final_output}")

if __name__ == "__main__":
    asyncio.run(run_agent_workflow())
