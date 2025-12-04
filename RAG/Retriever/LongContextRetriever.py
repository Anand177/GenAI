from langchain_chroma import Chroma
from langchain_community.document_transformers import LongContextReorder
from langchain_huggingface import HuggingFaceEmbeddings


data = [
    "RAG retrieves relevant documents to inform generation.",
    "RAG is used in open-domain question answering.",
    "The retrieval component in RAG finds pertinent information quickly.",
    "RAG can handle complex queries with more precision.",

    "Old t-shirts make great cleaning rags.",
    "Rags are perfect for dusting furniture.",
    "Use rags to clean up spills quickly.",
    "Cut up old towels for durable rags.",
    "Rags can be reused multiple times.",
    "Keep rags handy in the kitchen for quick cleanups.",
    "Rags made from cotton are highly absorbent.",
    "Store rags in a bucket for easy access.",
    "Rags are useful for polishing shoes.",
    "Recycle old clothes into rags instead of throwing them away."
]

# Creating Chroma DB and add documents
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store=Chroma.from_texts(data, embedding=embedding_function)
vector_store_retriever=vector_store.as_retriever(search_kwargs={"k": 10})

# Create Long Context Retriever
lc_retriever=LongContextReorder()

query="What is GPT and RAG"

#Get similar entries from Chroma DB
print("Original Ordering")
vc_results=vector_store_retriever.invoke(input=query)
for result in vc_results:
    print(result.page_content)

print("Long Context Retriever Ordering")
results=lc_retriever.transform_documents(vc_results)
for result in results:
    print(result.page_content)
