import asyncio
import os

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

sync=False

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

def sync_batch(llm, messages):

    batch_responses = llm.batch(messages)
    print("Responses received from Gemini Batch:")
    for i, (message, response) in enumerate(zip(messages, batch_responses)):
        print(f"Question {i}: {message[1].content}")
        print(f"Answer: {response.content}")
    return


async def async_batch(llm, messages):

    batch_responses = await llm.abatch(messages)
    print("Responses received from Gemini Async Batch:")
    for i, (message, response) in enumerate(zip(messages, batch_responses)):
        print(f"Question {i}: {message[1].content}")
        print(f"Answer: {response.content}")
    return


if __name__ == "__main__":
    print("Hello from Gemini Batch")

    systemMsg = SystemMessage("You are an assistant. Answer in no more than 40 words and no empty lines.")

    messages : list=[
        [ systemMsg, HumanMessage("Explain Gemini AI")],
        [ systemMsg, HumanMessage("When were you last updated?")]
    ]

    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7
    )
    if sync:
        sync_batch(llm, messages)
    else:
        asyncio.run(async_batch(llm, messages))
