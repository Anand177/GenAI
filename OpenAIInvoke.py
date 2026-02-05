from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAI

import os

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

print("Response 1 Start")
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
response= llm.invoke("When were you last updated")
print(response.content)
print("Response 1 End")

response= llm.invoke("Who is the PM of UK")
print(response.content)



"""
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, max_completion_tokens=20)
response= llm.invoke("Write a poem about sun in english for minimum of 8 lines and not more than 100 words")
print(response.content)
"""