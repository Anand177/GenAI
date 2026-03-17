from agents import Agent, function_tool
from dotenv import load_dotenv
from typing import Dict

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")

@function_tool
def send_email(subject: str, html_body: str) -> Dict[str, str]:
    """ Mock function to print mail """
    print(f"Subject -> {subject}")
    print(f"Html Body:\n{html_body}")
    return {"status" : "success" }

instructions = """You are able to send a nicely formatted HTML email based on a detailed report.
You will be provided with a detailed report. You should use your tool to send one email, providing the 
report converted into clean, well presented HTML with an appropriate subject line."""

email_agent = Agent(name="Email agent", instructions=instructions, tools=[send_email], model="gpt-4o-mini")
