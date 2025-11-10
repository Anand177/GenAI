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

textToCompare = {
    "source_sentence": "The quick brown fox jumps over the lazy dog.",
    "sentences": [
        "A fast dark-colored fox leaps over a sleepy canine.",
        "An apple a day keeps the doctor away.",
        "The rain in Spain stays mainly in the plain."
    ]
}


result = client.sentence_similarity(
    "The quick brown fox jumps over the lazy dog.",
     other_sentences=[
         "A fast dark-colored fox leaps over a sleepy canine.",
         "An apple a day keeps the doctor away.",
         "The rain in Spain stays mainly in the plain."
     ],
     model="google/embeddinggemma-300m")

print(result)

for txt in result:
    print(f"Sentence: {txt['sentence']} => Similarity Score: {txt['similarity_score']}")

