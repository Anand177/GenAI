from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import DocumentCompressorPipeline, EmbeddingsFilter
from langchain_classic.retrievers.document_compressors import LLMChainExtractor, LLMChainFilter
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os


# Utility Functions
def print_documents(docs):
    for i, doc in enumerate(docs):
        print(f"# {i}")
        print(doc.page_content)

def dump_before_after_compression(base_retriever, compressor, query):
    results_before=base_retriever.invoke(query)
    print(f"Before Document Count -> {len(results_before)}")
    print("-"*50)
    print_documents(results_before)
    print("-"*50)

    results_after=compressor.invoke(query)
    print(f"After Document Count -> {len(results_after)}")
    print("-"*50)
    print_documents(results_after)
    print("-"*50)

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

# Create LLM
llm=GoogleGenerativeAI(model="gemini-flash-latest",
                google_api_key=GOOGLE_API_KEY,
                temperature = 0.7,
                top_p = 0.65
)


# Setup Base Retriever
urls= [
    "https://en.wikipedia.org/wiki/Large_language_model",
    "https://en.wikipedia.org/wiki/Foundation_model",
    "https://en.wikipedia.org/wiki/Cache_language_model",
    "https://en.wikipedia.org/wiki/Generative_pre-trained_transformer",
    "https://en.wikipedia.org/wiki/Natural_language_processing",
    "https://en.wikipedia.org/wiki/Gemini_(language_model)",
    "https://en.wikipedia.org/wiki/BERT_(language_model)",
    "https://en.wikipedia.org/wiki/OpenAI",
    "https://en.wikipedia.org/wiki/ChatGPT",
    "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
]
loader=WebBaseLoader(web_paths=urls)
docs=loader.load()

doc_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
chunked_documents=doc_splitter.split_documents(docs)

embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store=Chroma(collection_name="RAG", embedding_function=embedding_function)
vector_store.add_documents(chunked_documents)
vector_store_retriever=vector_store.as_retriever()


query = "What is language model?"


# LLM Chain Extractor  --> Use LLm to extract relevant parts of Document
# Use LLM to extract relevant parts
llm_chain_extract_compressor=LLMChainExtractor.from_llm(llm=llm)
llm_chain_extract_compressor_retriever=ContextualCompressionRetriever(
    base_retriever=vector_store_retriever,
    base_compressor=llm_chain_extract_compressor
)
#dump_before_after_compression(vector_store_retriever, llm_chain_extract_compressor_retriever, query)


# LLM Filter  --> Drops irrelevant documents for query
#Use LLM to retain only relevant docs
llm_chain_filter_compressor=LLMChainFilter.from_llm(llm=llm)
llm_chain_filter_compressor_retriever=ContextualCompressionRetriever(
    base_retriever=vector_store_retriever,
    base_compressor=llm_chain_filter_compressor
)
#dump_before_after_compression(vector_store_retriever, llm_chain_filter_compressor_retriever, query)


# Embeddings Filter  --> Embed Document and find relevant document without LLM
similarity_threshold = 0.3 # Range 0 to 1.0|0 -> Max diversity|1 -> Similarity
embeddings_filter=EmbeddingsFilter(embeddings=HuggingFaceEmbeddings(),
                    similarity_threshold=similarity_threshold )
llm_embeddings_filter_compression_retriever= ContextualCompressionRetriever(
    base_retriever=vector_store_retriever,
    base_compressor=embeddings_filter
)
#dump_before_after_compression(vector_store_retriever, llm_embeddings_filter_compression_retriever, query)


# Compressor Pipeline  --> Use pipeline of transformers
transformers = [embeddings_filter, llm_chain_extract_compressor]
pipeline_compressor=DocumentCompressorPipeline(transformers=transformers)
pipeline_compressor_retriever= ContextualCompressionRetriever(
    base_retriever=vector_store_retriever,
    base_compressor=pipeline_compressor
)
dump_before_after_compression(vector_store_retriever, pipeline_compressor_retriever, query)