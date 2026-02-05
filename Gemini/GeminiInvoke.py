from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

import os

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

prompt="""
Predict the next word in the below sentence 

Sentence:
Fishing is

Output:
List of top 25 words predicted and their probabilities
"""

llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.3)
response = llm.invoke(prompt)
print(response.content)



llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY, temperature=2.0)
response = llm.invoke(prompt)
print(response.content)

