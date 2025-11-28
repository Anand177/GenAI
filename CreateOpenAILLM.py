from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

import os

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini",
            temperature=0.5,
            top_p=0.7,
            api_key=OPENAI_API_KEY,
            frequency_penalty=1
)

#print(llm)

response =llm.invoke("Who is the PM of England.")
print(response)