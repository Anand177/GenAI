import os

from dotenv import load_dotenv
from langchain_community.llms. import AI21
from groq import Groq

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

llm=AI21(
    model="j2-jumbo-instruct",
    ai21_api_key=A1I21_API_KEY,
    temperature=0.7,
)

print(llm.invoke("Explain AI21 LLM"))

