import os
import sys

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

template_text = "Tell me about {topic} in less than 100 words."
prompt_template = PromptTemplate( template=template_text, input_variables=["topic"] )

llm = GoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7)

test_chain = prompt_template | llm

response = test_chain.invoke({"topic": "LangChain"})
print("Response from Simple LCE:\n", response)

response = test_chain.invoke({"topic": "Gen AI"})
print("Response from Simple LCE:\n", response)


