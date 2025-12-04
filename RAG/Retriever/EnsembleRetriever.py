#pip install rank_bm25
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

corpus = [
    "RAG addresses hallucinations",
    "Symptoms are hallucinations",
    "RAG is easier than fine tuning",
    "Use a RAG to clean it",
    "Retrieval Augmented Generation",
    "RAG slows down performance of system due to addition of additional retriever step",
    "RAG is cheap to implement than Fine Tuning"
]

# Indexing Document
print("")
print("Corpus Document")
corpus_docs=[]
for i, dat in enumerate(corpus):
    document=Document(
        page_content=dat,
        metadata= {"doc_num" : i, "source": "Document # "+str(i)}
    )
    corpus_docs.append(document)

print(corpus_docs)

# Creating BM25 Retriever and add documents
bm25_retriever = BM25Retriever.from_documents(documents=corpus_docs, k=3)

# Creating Chroma DB and add documents
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store=Chroma(collection_name="ensemble_try", embedding_function=embedding_function)
vector_store.add_documents(corpus_docs)

chromadb_retriever = vector_store.as_retriever(search_kwargs={"k": 3})


#Create Ensemble Retriever
retriever_array=[bm25_retriever, chromadb_retriever]
retriever_weight=[0.4, 0.6]

ensemble_retriever = EnsembleRetriever(
    retrievers=retriever_array,
    weights=retriever_weight,
    id_key="source"
)

input=[
    "rag is cheap",
    "benefit of rag"
]

index=0
print(f"Question -> {input[index]}")

# BM25 Ranked list
print("BM25 Ranked List")
bm25_result=bm25_retriever.invoke(input[index])
print(bm25_result)

# Chroma Ranked list
print("BM25 Ranked List")
chroma_result=chromadb_retriever.invoke(input[index])
print(chroma_result)

# Ensemble Ranked list
print("Ensemble Retriever Ranked List")
ensemble_result=ensemble_retriever.invoke(input[index])
print(ensemble_result)
