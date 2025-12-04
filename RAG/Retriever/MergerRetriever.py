from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever, MergerRetriever
from langchain_classic.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_transformers import EmbeddingsClusteringFilter
from langchain_community.retrievers import WikipediaRetriever
from langchain_google_genai import GoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")


def print_documents(docs):
    for i, doc in enumerate(docs):
        print("#",i)
        print(doc.page_content)

def dump_results_info(result):
    print("Doc count = ", len(result))
    page_content_length=0
    for doc in result:
        page_content_length = page_content_length + len(doc.page_content)
    print("Context size = ", page_content_length)
    print_documents(result)


# Create Gemini LLM
llm=GoogleGenerativeAI(model="gemini-flash-latest",
                    google_api_key=GOOGLE_API_KEY,
                    temperature = 0.7,
                    top_p = 0.65
)

# Setting 3 Retrievers
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

urls= [
    "https://en.wikipedia.org/wiki/Large_language_model",
    "https://en.wikipedia.org/wiki/Foundation_model"
    "https://en.wikipedia.org/wiki/Generative_pre-trained_transformer"
    "https://en.wikipedia.org/wiki/Gemini_(language_model)",
    "https://en.wikipedia.org/wiki/BERT_(language_model)",
    "https://en.wikipedia.org/wiki/OpenAI",
    "https://en.wikipedia.org/wiki/ChatGPT",
    "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
]
loader=WebBaseLoader(web_paths=urls)
docs=loader.load()


# Retriever 1 -> ChromaDB with 200 chunk size and Similarity Search
doc_splitter_1=RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunked_docs_1=doc_splitter_1.split_documents(docs)

vector_store_1=Chroma(collection_name="Retriever_1", embedding_function=embedding_function)
vector_store_1.add_documents(chunked_docs_1)

vector_store_retriever_1=vector_store_1.as_retriever(search_type="similarity", 
                                search_kwargs={"k": 5})


# Retriever 2 -> ChromaDB with 500 chunk size and MMR
doc_splitter_2=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunked_docs_2=doc_splitter_2.split_documents(docs)

vector_store_2=Chroma(collection_name="Retriever_2", embedding_function=embedding_function)
vector_store_2.add_documents(chunked_docs_2)

vector_store_retriever_2=vector_store_2.as_retriever(search_type="mmr", 
                                search_kwargs={"k": 5})


# Retriever 3 -> Wikipedia Retriever
wikipedia_retriever=WikipediaRetriever(search_kwargs={"k": 5})

# Combine all three with merge Retriever
merger_retriever=MergerRetriever(retrievers=[vector_store_retriever_1,
                    vector_store_retriever_2, wikipedia_retriever])
# Create embedding clustering filter
filter_ordered_by_retriever= EmbeddingsClusteringFilter(embeddings=HuggingFaceEmbeddings(),
                                num_clusters=5,
                                num_closest=1,
                                sorted=True)
# Create document compressor pipeline
pipeline=DocumentCompressorPipeline(transformers=[filter_ordered_by_retriever])

# Create Compression Retriever
compression_retriever=ContextualCompressionRetriever(base_compressor=pipeline,
                                base_retriever=merger_retriever)

query= "What is RAG in Gen AI"
before=merger_retriever.invoke(query)
dump_results_info(before)

after=compression_retriever.invoke(query)
dump_results_info(after)