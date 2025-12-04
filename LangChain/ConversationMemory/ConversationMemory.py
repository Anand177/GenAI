from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.chat_message_histories import mes

import os

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

llm = GoogleGenerativeAI(model="gemini-flash-latest",
            google_api_key=GOOGLE_API_KEY,
            temperature = 0.7)

