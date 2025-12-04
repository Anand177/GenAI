from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

JSON_path="C:/Learning/Python/GenAI/EmbeddingAndVectorDB/PersonalInfo.JSON"

# Create Embedding Function
model_name="sentence-transformers/all-MiniLM-L6-v2"
embedding_function = HuggingFaceEmbeddings(model_name=model_name,
                        model_kwargs={'trust_remote_code': True})

# Load JSON Documents
def metadata_func(record: dict, metadata: dict) -> dict:
    """Extract name and userid as metadata"""
    metadata["name"] = record.get("name", "")
    metadata["userid"] = record.get("userid", "")
    return metadata

loader=JSONLoader(file_path=JSON_path, 
                  jq_schema='.[]',
                  text_content=False,
                  metadata_func=metadata_func)
docs=loader.load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)  # Not needed for small docs
chunked_docs=text_splitter.split_documents(docs)

# Embed and store in Chroma DB
collection_name="Personal_JSON"
persist_directory="C:/Learning/chromaDB/personal_json"
vector_store=Chroma(collection_name=collection_name,
                    embedding_function=embedding_function,
                    persist_directory=persist_directory)

#vector_store.add_documents(chunked_docs)
vector_store.add_documents(chunked_docs)

all_records=vector_store.get()
print(f"Total records: {len(all_records['ids'])}\n")
for i, (id, doc, metadata) in enumerate(zip(
    all_records['ids'], 
    all_records['documents'], 
    all_records['metadatas']
)):
    print(f"Record {i+1}:")
    print(f"ID: {id}")
    print(f"Content: {doc}")
    print(f"Metadata: {metadata}")
    print("-" * 80)