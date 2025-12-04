from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

# langchain Classic is supposedly decommissioned. However latest packsges aren't
# working. Latest alternatives are commented below
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
"""
from langchain.chains import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
"""

import os

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

##Using URL Loaded by ChromaDbUrlLoader.py
model_name="sentence-transformers/all-MiniLM-L6-v2"
persist_directory="C:/Learning/chromaDB/blogs"

# Connect to Chroma DB collection
collection_name="GenAI_Wiki"
embedding_function = HuggingFaceEmbeddings(model_name=model_name,
                        model_kwargs={'trust_remote_code': True})
vector_store = Chroma(collection_name=collection_name, 
                      embedding_function=embedding_function,
                      persist_directory=persist_directory)
retriever=vector_store.as_retriever()

llm = GoogleGenerativeAI(model="gemini-flash-latest",
            google_api_key=GOOGLE_API_KEY,
            temperature = 0.7)

# Define Multi Query Retriever
multi_q_retriever = MultiQueryRetriever.from_llm(llm=llm, retriever=retriever)
"""
input_qs = [
    "What is RAG",
    "What is LLM",
    "What is Fine Tuning",
    "What is chunking"
    ]

for input in input_qs:
    print(f"Question -> {input}")
    results=multi_q_retriever.invoke(input)
"""    

#Retrieval Chain Creation
system_prompt="""
You are an assistant for question and answer task
Use following piece of retrieved context to answer the question
Keep three sentences max and answer concise

Context: {context}
"""

prompt= ChatPromptTemplate.from_messages([
    ("system" , system_prompt),
    ("human", "{input}")
    ]
)


qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(multi_q_retriever, qa_chain)

input="What is tokenization"

print(multi_q_retriever.invoke(input))

response= rag_chain.invoke({"input" : input})

print(f"Question -> {response['input']}")
print(f"Answer -> {response['answer']}")
