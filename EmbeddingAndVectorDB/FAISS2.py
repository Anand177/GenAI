from dotenv import load_dotenv

from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import os

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

loader = WikipediaLoader("Metallica", lang="en")
docs = loader.load()

print(len(docs))
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap=20)

split_docs = text_splitter.split_documents(docs)

print(len(split_docs))

embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004",
                    google_api_key=GOOGLE_API_KEY)
library=FAISS.from_documents(split_docs,embeddings)

q1 = "Who replaced Cliff Burton in Metallica"
ans = library.similarity_search(q1, k=3)

