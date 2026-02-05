from transformers import BertConfig, BertModel, BertTokenizer
import torch

bertBaseConfig = BertConfig.from_pretrained('bert-base-uncased')

print(bertBaseConfig)

baseBertModel = BertModel(config=bertBaseConfig)
#print(baseBertModel)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#print(tokenizer)

#inputText = "The Quick brown fox jumps over the lazy dog. "
inputText = "This is authored by Anand Vasantharajan sant hara jan"
print("Input Text: ", inputText)

#Split sentence into tokens (word tokenizer - Each word is considered as Token)
tokens = tokenizer.tokenize(inputText)
print("Tokens: ", tokens)

#Assign unique IDs to each token (Numerical representation of each token in vocabulary)
encodedText = tokenizer.encode(inputText)
print("Encoded Text: ", encodedText)

#Convert back the encoded text to human readable format based on the tokenizer vocabulary
decodedText = tokenizer.decode(encodedText)
print("Decoded Text: ", decodedText)

#Convert input text to tensor number in pylist (Not the datatype tensor)
tensorText = tokenizer(inputText)
print("Tensor Text: ", tensorText)

#Convert input text to tensor based on PyTorch ('pt' => PyTorch)
tensorText = tokenizer(inputText, return_tensors='pt') 
print("Tensor Text: ", tensorText)

with torch.no_grad():
    outputs = baseBertModel(**tensorText)

print("Model Outputs: ", outputs)

