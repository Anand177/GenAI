from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
import os
import sys

# Only for customizing model creation for Google Generative AI
custom_config = {
    # Controls randomness (0.0 is deterministic, 1.0 is default)
    "temperature": 0.5, 
    
    # Nucleus sampling threshold
    "top_p": 0.9, 
    
    # Top K tokens to consider
    "top_k": 40, 
    
    # List of strings to immediately halt generation
    "stop_sequences": ["\n\n", "### END"], 
    
    # Maximum number of tokens to generate
    "max_output_tokens": 1024
}


# Google AI API
# https://api.python.langchain.com/en/latest/llms/langchain_google_genai.llms.GoogleGenerativeAI.html
# pip install --upgrade --quiet  langchain-google-genai
# use https://generativelanguage.googleapis.com/v1beta/models?key=${GEMINI_API_KEY} to find the appropriate model

def create_google_llm(model='gemini-flash-latest', args={}):
    GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
#    llm = GoogleGenerativeAI(model=model,google_api_key=GOOGLE_API_KEY, **args)
# Only for customizing model creation for Google Generative AI
    llm = GoogleGenerativeAI(model=model,
                             google_api_key=GOOGLE_API_KEY,
                             temperature = custom_config.get("temperature"),
                             top_p = custom_config.get("top_p"),
                             top_k = custom_config.get("top_k"), 
                             max_output_tokens = custom_config.get("max_output_tokens"),
                               **args)

    return llm




# Create Google API Key @ https://aistudio.google.com/app/projects
load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
print(GOOGLE_API_KEY)



llm=create_google_llm(model="gemini-flash-latest")

print(llm)


query = "How LLM works"

response = llm.invoke( query)

print(response)
