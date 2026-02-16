from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain.evaluation import load_evaluator

import os

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
HF_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# 1. Create HuggingFace Inference Embedding
model_name = "sentence-transformers/all-mpnet-base-v2"
model = HuggingFaceEndpointEmbeddings(model=model_name, huggingfacehub_api_token=HF_API_KEY)
#print(model)

ip_doc = ["Industrial revolution changed the world",
            "Indian independence inspired many non violent movements in the world"]

embed_ip_sentence = model.embed_documents(ip_doc)
print(f"Vector/Embedding size -> {len(embed_ip_sentence[0])}")
print(f"Count -> {len(embed_ip_sentence)}")


# 2. Using Evaluator to calculate similarity
pairwise_distance_evaluator = load_evaluator("pairwise_embedding_distance", embeddings=model)

query = "Steam Engine invention completely changed travel"
print("Cosine Evaluation")
for ip in ip_doc:
    distance = pairwise_distance_evaluator.evaluate_string_pairs(prediction=query, prediction_b=ip, distance_metric="cosine")
    print(f"Distance -> {distance}\t Input -> {ip}")
"""
query = "Steam Engine invention completely changed travel"
print("manhattan Evaluation")
for ip in ip_doc:
    distance = pairwise_distance_evaluator.evaluate_string_pairs(prediction=query, prediction_b=ip, distance_metric="manhattan")
    print(f"Distance -> {distance}\t Input -> {ip}")

query = "Steam Engine invention completely changed travel"
print("Euclidean Evaluation")
for ip in ip_doc:
    distance = pairwise_distance_evaluator.evaluate_string_pairs(prediction=query, prediction_b=ip, distance_metric="euclidean")
    print(f"Distance -> {distance}\t Input -> {ip}")
"""