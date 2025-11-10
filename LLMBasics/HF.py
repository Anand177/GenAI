from dotenv import load_dotenv
import sys
import os
from huggingface_hub import InferenceClient


load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
sys.path.append('../')

HF_API_KEY=os.getenv("HUGGINGFACEHUB_API_TOKEN")
print(HF_API_KEY)


client = InferenceClient(
    provider="auto",
    api_key=HF_API_KEY,
)

completion = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V3.1-Terminus",
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
)

print(completion.choices[0].message.content)