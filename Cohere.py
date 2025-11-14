from dotenv import load_dotenv
import sys
import os

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
sys.path.append('../')
COHERE_API_KEY=os.getenv("COHERE_API_KEY")

print(COHERE_API_KEY)