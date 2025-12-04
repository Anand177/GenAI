from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# langchain Classic is supposedly decommissioned. However latest packsges aren't
# working. Latest alternatives are commented below
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
"""
from langchain.chains import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
"""

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

retriever=vector_store.as_retriever()

llm = ChatOpenAI(model="gpt-4o-mini",
            temperature=0.5,
            top_p=0.7,
            api_key=OPENAI_API_KEY,
            frequency_penalty=1
)



#Set Prompts
ques_ans_system_prompt="""
    Given chat history and latest user question which may refer context in chat history,
    formulate a standalone question which can be understood without the chat history. 
    Do NOT answer the question, just rephrase it if needed and or return it as is."
"""

ques_ans_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ques_ans_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

hist_retrieve = create_history_aware_retriever(llm, retriever, ques_ans_prompt)

system_prompt="""
You are an assistant for question and answer task
Use following piece of retrieved context to answer the question
Keep three sentences max and answer concise

Context: {context}
"""

prompt= ChatPromptTemplate.from_messages(
    [
        ("system" , system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

qa_chain=create_stuff_documents_chain(llm=llm, prompt=prompt)
rag_chain= create_retrieval_chain(hist_retrieve, qa_chain)


chat_history = []
def  invoke_llm(input):
    response = rag_chain.invoke({"input": input, "chat_history": chat_history})
    chat_history.extend(
        [
            HumanMessage(content=input),
            AIMessage(content=response["answer"]),
        ]
    )
    return response

response = invoke_llm("What is LLM")
print(response['answer'])

response = invoke_llm("How is it different from Foundation Model")
print(response['answer'])
