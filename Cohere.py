from dotenv import load_dotenv

from langchain_cohere import ChatCohere
import os

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
COHERE_API_KEY=os.getenv("COHERE_API_KEY")

print(COHERE_API_KEY)

llm = ChatCohere(cohere_api_key=COHERE_API_KEY, model="command-a-03-2025")
response =llm.invoke("tell me about model being invoked")
print(response)
