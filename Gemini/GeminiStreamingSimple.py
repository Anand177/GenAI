import asyncio
import os

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from dotenv import load_dotenv

invoke=False
ainvoke=False
stream=True

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

llm = GoogleGenerativeAI(model="gemini-flash-latest", google_api_key=GOOGLE_API_KEY)
query="What is LLM"

if invoke:
    response=llm.invoke(query)
    print(response)
if ainvoke:
    stream_chunks=llm.stream(query)
    for chunk in stream_chunks:
        print(chunk, end='', flush=True)


async def llm_async_call():
    print("Starting Async Call")
    response = await llm.ainvoke(query)
    print("Receiving Response")
    print(response)


if stream:
    asyncio.run(llm_async_call())

"""
def sync_stream(llm, message):

    chunk = llm.stream(message)
    print("Response received from Gemini Streaming:")
    for part in chunk:
        print(part.content, end='', flush=True)
    return

async def async_stream(llm, message):

    chunk = llm.astream(message)
    print("Response received from Gemini Async Streaming:")
    async for part in chunk:
        print(part.content, end='', flush=True)
    return


if __name__ == "__main__":
    print("Hello from Gemini Streaming")

    message : list=[
        SystemMessage("You are an assistant. Answer in detailed and informative way.")
    ]

    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7
    )
    message.append(HumanMessage("Explain Gemini AI"))

    if sync:
        sync_stream(llm, message)
    else:
        asyncio.run(async_stream(llm, message))
"""