import asyncio
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

async def main():
    
    message=[
        {"role": "system", "content": "You are an assistant. Answer in no more than 100 words."}
    ]
    
    load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7
    )

    task1 = asyncio.create_task(llm.ainvoke(
        message + [{"role": "user", "content": "Explain Gemini AI in brief."}]
    ))
    task2 = asyncio.create_task(llm.ainvoke(
        message + [{"role": "user", "content": "When model last updated?"}]
    ))
    task3 = asyncio.create_task(llm.ainvoke(
        message + [{"role": "user", "content": "What are the key features of Gemini AI"}]
    ))

    print("Requests sent, waiting for responses...")
    print("Still waiting...")

    response = await asyncio.gather(task1, task2, task3) 
    print("Response received from Gemini Async:")
    for res in response:
        print(res.content)
        print(res)
    

if __name__ == "__main__":
    asyncio.run(main())
    print("Hello from Gemini Main")