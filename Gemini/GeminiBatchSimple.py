from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI

import asyncio, os

sync=False
load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

def sync_batch(llm, messages):

    batch_responses = llm.batch(messages)
    print("Responses received from Gemini Batch:")
    for i, (message, response) in enumerate(zip(messages, batch_responses)):
        print(f"Question {i}: {message}")
        print(f"Answer: {response}")
    return


async def async_batch(llm, messages):

    batch_responses = await llm.abatch(messages)
    print("Responses received from Gemini Async Batch:")
    for i, (message, response) in enumerate(zip(messages, batch_responses)):
        print(f"Question {i}: {message}")
        print(f"Answer: {response}")
    return


if __name__ == "__main__":

    messages : list=[
        "Explain Gemini AI in no more than 50 words and no empty lines",
        "When were you last updated?"
    ]

    llm = GoogleGenerativeAI(model="gemini-flash-latest", google_api_key=GOOGLE_API_KEY)
    if sync:
        sync_batch(llm, messages)
    else:
        asyncio.run(async_batch(llm, messages))
