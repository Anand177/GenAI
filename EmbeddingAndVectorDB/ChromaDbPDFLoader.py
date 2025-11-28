from langchain_community.document_loaders import PyPDFLoader
#from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

source_pdf = "C:/Users/anand/Downloads/Policy_POPM2W00103407164.pdf"

pdf_loader =PyPDFLoader(source_pdf)
pdf_document = pdf_loader.load()

print(f"Total Docs -> {len(pdf_document)}")

print(f"MetaData -> {pdf_document[0].metadata}")
#print(f"Page Content -> {pdf_document[0].page_content}")


chunk_size=512 #max chunk size
chunk_overlap=100 # 100 tokens can be found in overlapping chunks

pdf_text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap,
                keep_separator=False,
                )

chuncked_doc=pdf_text_splitter.split_documents(pdf_document)
print(f"Total Chunks -> {len(chuncked_doc)}")

print("Chunked Doc 0 :")
print("-----------------------")
print(chuncked_doc[0].page_content)
print("-----------------------")
print("Chunked Doc 1 :")
print("-----------------------")
print(chuncked_doc[1].page_content)
print("-----------------------")

print(f"Chunk 0 Length -> {len(chuncked_doc[0].page_content)}")
print(f"Chunk 1 Length -> {len(chuncked_doc[1].page_content)}")

model_name="sentence-transformers/all-MiniLM-L6-v2"
persist_directory="C:/Learning/chromaDB/PolicyDoc"

collection_name="Anand_vehicle_policy"
collection_metadata={"embedding":model_name, "chunk_size": chunk_size, 
                "chunk_overlap": chunk_overlap}

embedding_function = HuggingFaceEmbeddings(model_name=model_name,
                        model_kwargs={'trust_remote_code': True})
vector_store = Chroma(collection_name=collection_name, 
                      collection_metadata=collection_metadata,
                      embedding_function=embedding_function,
                      persist_directory=persist_directory)

vector_store.add_documents(chuncked_doc)

k=3

query = "What is policy number"
results = vector_store.similarity_search(query=query, k=k)

for result in results:
    print(f"Metadata -> {result.metadata}")



## MMR Search
# Range 0 to 1. 0 for max diversity and 1 for similarity
lambda_mult=0.5 
mmr_results=vector_store.max_marginal_relevance_search(query=query, k=k,
                lambda_mult=lambda_mult)
for result in mmr_results:
    print(f"Metadata -> {result.metadata}")
