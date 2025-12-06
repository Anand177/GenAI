from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

import os

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

##Using URL Loaded by ChromaDbUrlLoader.py

model_name="sentence-transformers/all-MiniLM-L6-v2"
persist_directory="C:/Learning/chromaDB/blogs"

# Define Chroma DB collection
collection_name="GenAI_Wiki"
embedding_function = HuggingFaceEmbeddings(model_name=model_name,
                        model_kwargs={'trust_remote_code': True})
vector_store = Chroma(collection_name=collection_name, 
                      embedding_function=embedding_function,
                      persist_directory=persist_directory)

#Set Prompts
system_prompt="""
You are an assistant for question and answer task
Use following piece of retrieved context to answer the question
Keep three sentences max and answer concise

Context: {context}
"""

prompt= ChatPromptTemplate.from_messages(
    [
        ("system" , system_prompt),
        ("human", "{input}")
    ]
)



retriever=vector_store.as_retriever()

llm = ChatOpenAI(model="gpt-4o-mini",
            temperature=0.5,
            top_p=0.7,
            api_key=OPENAI_API_KEY,
            frequency_penalty=1
)

qa_chain=create_stuff_documents_chain(llm=llm, prompt=prompt)
rag_chain= create_retrieval_chain(retriever, qa_chain)

response = rag_chain.invoke({"input" : "What is LLM"})
print(response['answer'])

response = rag_chain.invoke({"input" : "How is it different from Foundation Model"})
print(response['answer'])