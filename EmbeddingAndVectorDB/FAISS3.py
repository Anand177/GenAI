from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import faiss
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")

model_name="models/gemini-embedding-001"
embeddings_model = GoogleGenerativeAIEmbeddings(model=model_name)

def print_query_result(index_faiss):
    
    test_docs = [
        "Who killed Padmé",
        "Who is Anakin Skywalker",
        "Who is Jack Sparrow"
    ]

    
    k=3 

    for doc in test_docs:

        print(f"Doc to query -> {doc}")
        embed_query= embeddings_model.embed_documents([doc])

        distances, indexes = index_faiss.search(np.array(embed_query), k)
        print(f"Distance -> {distances}")
        print(f"Index -> {indexes}")

        for i, inp_index in enumerate(indexes[0]):
            print(f"Nearest Sentance -> {split_docs[inp_index]} | Score -> {distances[0][i]}")


loader = WebBaseLoader(web_paths=[
    "https://en.wikipedia.org/wiki/Star_Wars",
    "https://en.wikipedia.org/wiki/Star_Wars:_Episode_I_%E2%80%93_The_Phantom_Menace",
    "https://en.wikipedia.org/wiki/Star_Wars:_Episode_II_%E2%80%93_Attack_of_the_Clones",
    "https://en.wikipedia.org/wiki/Star_Wars:_Episode_III_%E2%80%93_Revenge_of_the_Sith",
    "https://en.wikipedia.org/wiki/Star_Wars_(film)",
    "https://en.wikipedia.org/wiki/The_Empire_Strikes_Back",
    "https://en.wikipedia.org/wiki/Return_of_the_Jedi",
    "https://en.wikipedia.org/wiki/Star_Wars:_The_Force_Awakens",
    "https://en.wikipedia.org/wiki/Star_Wars:_The_Last_Jedi",
    "https://en.wikipedia.org/wiki/Star_Wars:_The_Rise_of_Skywalker"
])
docs = loader.load()

print(f"Doc Length -> {len(docs)}")
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap=20)

split_docs = text_splitter.split_documents(docs)
print(f"Split Doc length -> {len(split_docs)}")

doc_texts = [doc.page_content for doc in split_docs]
print(f"Number of text chunks -> {len(doc_texts)}")

#Embedding list
doc_embeddings = embeddings_model.embed_documents(doc_texts)
print(f"First Embedding -> {doc_embeddings[0][:10]}")
doc_embeddings_numpy = np.array(doc_embeddings).astype(np.float32)
print(doc_embeddings_numpy)

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
nbits = 768 # should be multiple of dimension size of hash bit

index_lsh = faiss.IndexLSH(embedding_dimension, nbits)
index_lsh.add(doc_embeddings_numpy)

print("Index LSH")
print_query_result(index_lsh)