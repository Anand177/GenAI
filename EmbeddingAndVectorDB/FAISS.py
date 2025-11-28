from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import faiss
import numpy as np

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")

model_name="models/text-embedding-004"
embeddings_model = GoogleGenerativeAIEmbeddings(model=model_name)

def print_query_result(index_faiss):
    
    test_docs = [
        'I am a foodie',
        'My siter loves to play string instruments',
        'Musical instruments'
    ]

    
    k=3 

    for doc in test_docs:

        print(f"Doc to query -> {doc}")
        embed_query= embeddings_model.embed_documents([doc])

        distances, indexes = index_faiss.search(np.array(embed_query), k)
        print(f"Distance -> {distances}")
        print(f"Index -> {indexes}")

        for i, inp_index in enumerate(indexes[0]):
            print(f"Nearest Sentance -> {doc_to_index[inp_index]} | Score -> {distances[0][i]}")


doc_to_index = [
  "A man is eating food.", "A man is eating a piece of bread.",
  "The chef is preparing a delicious meal in the kitchen.", "A chef is tossing vegetables in a sizzling pan.",
  "A man is riding a horse.", "A man is riding a white horse on an enclosed ground.",
  "A woman is playing violin.", "A musician is tuning his guitar before the concert.",
  "The girl is carrying a baby.", "The baby is giggling while playing with her toys.",
  "The family is having a picnic under the shady oak tree.", "A group of friends is hiking up the mountain trail.",
  "The mechanic is repairing a broken-down car in the garage.", "The old man is feeding breadcrumbs to the ducks at the pond.",
  "The artist is sketching a beautiful landscape at sunset.", "A man is painting a colorful mural on the city wall.",
  "A team of scientists is conducting experiments in the laboratory.", "A group of students is studying together in the library.",
  "The birds are chirping happily in the morning sun.", "The dog is chasing its tail around the backyard.",
  "A group of children are playing soccer in the park.", "A monkey is playing drums.",
  "A boy is flying a kite in the open field.", "Two men pushed carts through the woods.",
  "A woman is walking her dog along the beach.", "A young girl is reading a book under a shady tree.",
  "The dancer is gracefully performing on stage.", "The farmer is harvesting ripe tomatoes from the vine."
]

#Embedding list
doc_embeddings = embeddings_model.embed_documents(doc_to_index)
doc_embeddings_numpy = np.array(doc_embeddings).astype(np.float32)

embedding_dimension = len(doc_embeddings[0])

print("Embedding dimension = ", embedding_dimension)


# FAISS flatL2 indexing
# create index with dimension of embedding model's vecor size
index_flatl2=faiss.IndexFlatL2(embedding_dimension)

print(f"Is trained -> {index_flatl2.is_trained}")

# add embeddings
index_flatl2.add(doc_embeddings_numpy)
print(f"Size of Index -> {index_flatl2.ntotal}")

print("index_flatl2")
# query index
print_query_result(index_flatl2)


# Index LSH -> Hash and group similar vectors together
# Higher better recall but low QPS
nbits = 32 # should be multiple of dimension size of hash bit

index_lsh = faiss.IndexLSH(embedding_dimension, nbits)
index_lsh.add(doc_embeddings_numpy)

print("Index LSH")
print_query_result(index_lsh)