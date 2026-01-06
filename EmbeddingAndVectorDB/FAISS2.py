from dotenv import load_dotenv

from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI

import os

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

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

print(len(docs))
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap=20)

split_docs = text_splitter.split_documents(docs)

print(len(split_docs))

embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004",
                    google_api_key=GOOGLE_API_KEY)
library=FAISS.from_documents(split_docs,embeddings)

q1 = "Who killed Padmé"
print(f"Question -> {q1}")
ans = library.similarity_search(q1, k=3)

print(ans)
#print(ans[0].page_content)
#print(ans[1].page_content)

ans = library.similarity_search_with_score(q1, k=3)
print(ans)

q1 = "Who is Anakin Skywalker"
ans = library.similarity_search(q1, k=3)
print(f"Question -> {q1}")
print(ans)


llm = GoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0.7
)

retriever = library.as_retriever()
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

retriever_q = "What is the most successful Starwars movie"
print(f"Question -> {retriever_q}")
results = qa.invoke(retriever_q)
print(results)

retriever_q = "Who killed Padmé"
print(f"Question -> {retriever_q}")
results = qa.invoke(retriever_q)
print(results)

retriever_q = "Who is Anakin Skywalker"
print(f"Question -> {retriever_q}")
results = qa.invoke(retriever_q)
print(results)


retriever_q = "Who is jack Sparrow"
print(f"Question -> {retriever_q}")
results = qa.invoke(retriever_q)
print(results)

"""
library.save_local("faiss_index")
saved_lib = FAISS.load_local("faiss_index", embeddings)
"""
