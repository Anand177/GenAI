from dotenv import load_dotenv

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

loader = WebBaseLoader(web_paths=[
    "https://en.wikipedia.org/wiki/Star_Wars",
    "https://en.wikipedia.org/wiki/Star_Wars:_Episode_I_%E2%80%93_The_Phantom_Menace",
    "https://en.wikipedia.org/wiki/Star_Wars:_Episode_II_%E2%80%93_Attack_of_the_Clones",
    "https://en.wikipedia.org/wiki/Star_Wars:_Episode_III_%E2%80%93_Revenge_of_the_Sith",
    "https://en.wikipedia.org/wiki/Star_Wars_(film)",
    "https://en.wikipedia.org/wiki/The_Empire_Strikes_Back",
    "https://en.wikipedia.org/wiki/Return_of_the_Jedi",
    "https://en.wikipedia.org/wiki/Star_Wars:_The_Force_Awakens",
    "https://en.wikipedia.org/wiki/Star_Wars:_The_Last_Jedi",
    "https://en.wikipedia.org/wiki/Star_Wars:_The_Rise_of_Skywalker"
])
docs = loader.load()

print(len(docs))
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap=20)

split_docs = text_splitter.split_documents(docs)

print(len(split_docs))

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001",
                    google_api_key=GOOGLE_API_KEY)
library=FAISS.from_documents(split_docs,embeddings)

q1 = "Who killed Padmé"
print(f"Question -> {q1}")
ans = library.similarity_search(q1, k=3)

print(ans)
#print(ans[0].page_content)
#print(ans[1].page_content)

ans = library.similarity_search_with_score(q1, k=3)
print(ans)

q1 = "Who is Anakin Skywalker"
ans = library.similarity_search(q1, k=3)
print(f"Question -> {q1}")
print(ans)


llm = GoogleGenerativeAI(model="gemini-flash-latest", temperature=0.7)
system_prompt=("Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

retriever = library.as_retriever()
question_answer_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(retriever, question_answer_chain)

retriever_q = "What is the most successful Starwars movie"
print(f"Question -> {retriever_q}")
results = qa_chain.invoke({"input": retriever_q})
print(results)

retriever_q = "Who killed Padmé"
print(f"Question -> {retriever_q}")
results = qa_chain.invoke({"input": retriever_q})
print(results)

retriever_q = "Who is Anakin Skywalker"
print(f"Question -> {retriever_q}")
results = qa_chain.invoke({"input": retriever_q})
print(results)


retriever_q = "Who is jack Sparrow"
print(f"Question -> {retriever_q}")
results = qa_chain.invoke({"input": retriever_q})
print(results)

"""
library.save_local("faiss_index")
saved_lib = FAISS.load_local("faiss_index", embeddings)
"""
