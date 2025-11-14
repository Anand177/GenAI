from dotenv import load_dotenv
import sys
import os
from huggingface_hub import InferenceClient


load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
sys.path.append('../')

HF_API_KEY=os.getenv("HUGGINGFACEHUB_API_TOKEN")
print(HF_API_KEY)


client = InferenceClient(
    provider="hf-inference",
    api_key=HF_API_KEY,
)

inp = """Using the triple quotes style is one of the easiest and most common ways to split a large string 
into a multiline Python string. 
Triple quotes can be used to create a multiline string. 
It allows you to format text over many lines and include line breaks. 
Put two triple quotes around the multiline Python string, one at the start and one at the end, to define it."""

result = client.summarization(inp, model="facebook/bart-large-cnn")

print(result)