from dotenv import load_dotenv
from agents import Agent, Runner, trace

import asyncio, os

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

agent=Agent(name="Jokester", instructions="You are a joke teller",  # System Prompt
            model="gpt-4o-mini")

print(agent)

async def agent_fn():
    result = await Runner.run(agent, "Tell joke on Open AI")      # Returns Co-routine
    print(result.final_output)

    with trace("Joke Telling"):
        """Creates Trace logs
        Visti: https://platform.openai.com/logs?api=traces"""
        result = await Runner.run(agent, "Tell joke on Open AI")
        print(result.final_output)

asyncio.run(agent_fn())