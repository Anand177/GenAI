from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore, LocalFileStore
from langchain_core.prompts import PromptTemplate
from langchain_core.load.serializable import s
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os

#Create Embedding Function
model_name="sentence-transformers/all-MiniLM-L6-v2"
embedding_function = HuggingFaceEmbeddings(model_name=model_name,
                        model_kwargs={'trust_remote_code': True})

llm_url="https://en.wikipedia.org/wiki/Large_language_model"
rag_url="https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
gemini_url="https://en.wikipedia.org/wiki/Gemini_(language_model)"
bert_url="https://en.wikipedia.org/wiki/BERT_(language_model)"

#Loading couple of Blogs
loader=WebBaseLoader(web_paths=(llm_url, rag_url, gemini_url, bert_url))
docs=loader.load()

chunk_size=300
chunk_overlap=50

child_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
parent_splitter=RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=20)

persist_directory="C:/Learning/chromaDB/blogs"
collection_name="GenAI_Wiki"

#Clear CHild storage to store fresh data
if os.path.exists(persist_directory):
    import shutil
    shutil.rmtree(persist_directory)

vector_store = Chroma(collection_name=collection_name, 
                      embedding_function=embedding_function,
                      persist_directory=persist_directory)

#parent_doc_store=InMemoryStore()

## Uncomment below three lines if using local memory for Parent Doc Store
PARENT_DIR_STORE="C:/Learning/chromaDB/parent_blogs"
os.makedirs(PARENT_DIR_STORE, exist_ok=True)
parent_doc_store=LocalFileStore(root_path=PARENT_DIR_STORE,)


parent_retriever = ParentDocumentRetriever(
    vectorstore=vector_store,   #Child Chunk
    docstore=parent_doc_store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter # Need not mention if full doc has to be added as Parent
)
# Need not add document to child chunks
parent_retriever.add_documents(docs)
print(f"Total Parent Documents stored: {len(parent_doc_store.store)}")

print("\nParent document IDs:")
#for key in list(parent_doc_store.store.keys())[:5]:  # Print first 5 keys
#    print(f"  {key}")

input_qs = [
    "What is RAG",
    "What is LLM",
    "What is Fine Tuning",
    "What is chunking"
    ]

q=2
query = input_qs[q]

print(f"Question -> {query}")
print("Retrieving from child ")
results = vector_store.as_retriever(search_kwargs={"k": 3}).invoke(query)

parent_docs=[]
parent_ids_seen=set()

for i, child_doc in enumerate(results):
    print(f"Child Chunk[{i}]")
    print(f"Child Metadata -> {child_doc.metadata}")
    print(f"Child Content -> {child_doc.page_content[:150]}")

    parent_id = child_doc.metadata.get('doc_id')
    if parent_id not in parent_ids_seen:
        parent_doc=parent_doc_store.mget([parent_id])[0]
        if parent_doc:
            print(f"Parent Document:")
            print(f"Source: {parent_doc.metadata.get('source')}")
            print(f"Content Length: {len(parent_doc.page_content)}")
            parent_docs.append(parent_doc)
            parent_ids_seen.add(parent_id)
        else:
            print(f"Parent document not found for ID: {parent_id}")
    else:
        print(f"No parent doc_id found in metadata")

print(f"Total Parent IDs -> {len(parent_ids_seen)}")
print(f"Total Parent Docs -> {len(parent_docs)}")

### Use below only to directly query from parent for debugging
"""
print("Retrieving from parent")
result_parent=parent_retriever.invoke(query)
# The result will be a list of parent Documents
for i, doc in enumerate(result_parent):
    print(f"Parent Doc[{i}]. Source: {doc.metadata.get('source')}")
    print(f"Parent Doc[{i}]. Content Length: {len(doc.page_content)}")
    print(f"Parent Doc[{i}]. Content (start): {doc.page_content[:200]}...")
"""

#Build Context from parent docs
context = "\n".join([doc.page_content  for doc in parent_docs])

instruction = """Use the following pieces of context to answer the question. 
If you don't knowr, just say that you don't know, don't make up an answer.

Context: {context}

Question: {query}
"""
prompt_template = PromptTemplate(
    template=instruction,
    input_variables=["context", "query"]
)
# Initiate LLM
load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

llm = GoogleGenerativeAI(model="gemini-flash-latest",
                             google_api_key=GOOGLE_API_KEY,
                             temperature = 0.7,
                             top_p = 0.65)

chain = prompt_template | llm

response = chain.invoke({"context": context, "query": query})
print(response)