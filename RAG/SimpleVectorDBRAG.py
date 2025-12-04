from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI
import os

# Reusing Data indexed by ChromaDBJSONLoader.py


# Initialize the same embedding function
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_function = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={'trust_remote_code': True}
)

# Connect to existing ChromaDB
collection_name = "Personal_JSON"
persist_directory = "C:/Learning/chromaDB/personal_json"
vector_store = Chroma(
    collection_name=collection_name,
    embedding_function=embedding_function,
    persist_directory=persist_directory
)

# Query for User 
user="anandvasantharajan" 
"""
result=vector_store.similarity_search(
    "name",
    k=3,
    filter={
        "$or": [{"userid" : user},
                {"name": user}
        ]
    }
)
print(f"Query Result ->{result}")
"""
retriever_filtered = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3,
        "filter":{
            "$or": [{"userid" : user},
                {"name": user}
            ]
        }
    }
)

print(f"Retriever Result ->{retriever_filtered}")
#context=""
#for doc in result:
#    context="".join(doc.page_content)
#print(f"Query Content ->{context}")

#Set Prompts
system_prompt="""
You are an assistant for question and answer task
User Information is being provided below
Provide recommendations and guidance to user from the context
If you dont know the answer, mention that you dont know and dont hallucinate

User Information: {context}
"""

prompt= ChatPromptTemplate.from_messages(
    [
        ("system" , system_prompt),
        ("human", "{input}")
    ]
)


# Initialize LLM
load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
llm=GoogleGenerativeAI(model="gemini-flash-latest",
                    google_api_key=GOOGLE_API_KEY,
                    temperature = 0.7,
                    top_p = 0.65
)

qa_chain=create_stuff_documents_chain(llm=llm, prompt=prompt)
rag_chain= create_retrieval_chain(retriever_filtered, qa_chain)

response=rag_chain.invoke({"input" : "What is my name, profession and where do I work"})
print(response['answer'])

response=rag_chain.invoke({"input" : "Generate a writeup not exceeding 200 words about my company"})
print(response['answer'])

response=rag_chain.invoke({"input" : "Recommend some activities/workshops for my hobbies"})
print(response['answer'])

response=rag_chain.invoke({"input" : "What all languages do I speak"})
print(response['answer'])

response=rag_chain.invoke({"input" : "Generate writeup on my telephone"})
print(response['answer'])