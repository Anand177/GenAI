from dotenv import load_dotenv
import sys
import os
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification  #pip install transformers
import torch    #pip install torch


load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
sys.path.append('../')

HF_TOKEN=os.getenv("HUGGINGFACEHUB_API_TOKEN")

model_name = "tabularisai/multilingual-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment_map = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}
    return [sentiment_map[p] for p in torch.argmax(probabilities, dim=-1).tolist()]

texts = [
    "I love programming!",
    "I hate bugs.", 
    "The weather is okay."
    "I love this product!",
    "This is the worst service ever.",
]


for text, sentiment in zip(texts, predict_sentiment(texts)):
    print(f"Text: {text} => Sentiment: {sentiment}")


'''simple process
client = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN,
)

result = client.text_classification(
    "I like you. I love you",
#    model="ProsusAI/finbert",  simple model with three levels of classification
    model="tabularisai/multilingual-sentiment-analysis", #advanced model with five levels of classification
)

print(result)

'''