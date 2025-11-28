from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

llm_url="https://en.wikipedia.org/wiki/Large_language_model"
rag_url="https://en.wikipedia.org/wiki/Retrieval-augmented_generation"

#Loading couple of Blogs
loader=WebBaseLoader(web_paths=(llm_url, rag_url))
docs=loader.load()

chunk_size=600
chunk_overlap=100

text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
chunked_docs=text_splitter.split_documents(docs)

print(f"Total Chunks -> {len(chunked_docs)}")
print(f"Chunk 0 Length -> {len(chunked_docs[0].page_content)}")
print(f"Chunk 1 Length -> {len(chunked_docs[1].page_content)}")

model_name="sentence-transformers/all-MiniLM-L6-v2"
persist_directory="C:/Learning/chromaDB/blogs"
collection_name="GenAI_Wiki"

embedding_function = HuggingFaceEmbeddings(model_name=model_name,
                        model_kwargs={'trust_remote_code': True})
vector_store = Chroma(collection_name=collection_name, 
                      embedding_function=embedding_function,
                      persist_directory=persist_directory)

vector_store.add_documents(chunked_docs)
