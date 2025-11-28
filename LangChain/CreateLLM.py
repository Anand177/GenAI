from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_huggingface.llms.huggingface_endpoint import HuggingFaceEndpoint
import os

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
HF_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")

print("Testing HuggingFace models...\n")

# Test multiple models
models = [
    "HuggingFaceH4/zephyr-7b-beta",
    "tiiuae/falcon-7b-instruct",
    "google/gemma-2-2b-it",
    "mistralai/Mistral-7B-Instruct-v0.2"
]

for model in models:
    print(f"Trying {model}...")
    try:
        llm_hf = HuggingFaceEndpoint(
            repo_id=model,
            huggingfacehub_api_token=HF_API_KEY,
            temperature=0.7,
            max_new_tokens=128,
            timeout=60
        )
        
        prompt = "Who are you?"
        response = llm_hf.invoke([{"role": "user", "content": prompt}])
        print(f"✓ SUCCESS with {model}")
        print(f"Response: {response}\n")
        break
        
    except Exception as e:
        print(f"✗ Failed: {str(e)[:100]}\n")
        continue