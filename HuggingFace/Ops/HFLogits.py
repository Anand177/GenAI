import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertConfig, DistilBertModel

modelName = "distilbert-base-uncased-finetuned-sst-2-english"

model = DistilBertConfig(modelName)

tokenizer = DistilBertTokenizer.from_pretrained(modelName)

inputText = "This python program is written by Anand Vasantharajan."
print("Input Text: ", inputText)

tokens = tokenizer(inputText, return_tensors="pt")

with torch.no_grad():
    outputs = model(**tokens)

print(type(outputs))
print(outputs)