from dotenv import load_dotenv
from huggingface_hub import InferenceClient

import os


load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
HF_TOKEN=os.getenv("HUGGINGFACEHUB_API_TOKEN")


client = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN
)

output = client.image_classification("C:/Users/anand/Pictures/Pics/D750/2020_02_28_Anvith_BDay/DSC_3570.JPG", 
                    model="google/vit-large-patch32-384")
print(output)