from dotenv import load_dotenv
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List

import json
import os
import uuid

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Learning/AI/Key/gen-lang-client-0398817262-d752d424f736.json"

project_id = "gen-lang-client-0398817262"
dataset_id = "GenAi"
table_id = "personal_json"

full_table_id = f"{project_id}.{dataset_id}.{table_id}"
VECTOR_MODEL = "text-embedding-004"

JSON_path = "C:/Learning/Python/GenAI/EmbeddingAndVectorDB/PersonalInfo.JSON"
embeddings_model = GoogleGenerativeAIEmbeddings(
        model=VECTOR_MODEL,
        google_api_key=GOOGLE_API_KEY
)


# Create BQ Table
def create_table(client : bigquery.Client):
    schema = [
        bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("content", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("metadata", "JSON", mode="REQUIRED"),
        bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED")
    ]
    table = bigquery.Table(full_table_id, schema)

    try:
        client.get_table(table=full_table_id)
        print(f"Table -> {full_table_id} already exists")
    except NotFound:
        print(f"Creating Table -> {full_table_id}")
        client.create_table(table=table)
        print(f"Table -> {full_table_id} successfully created")


#   Extract name and userid as metadata
def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["name"] = record.get("name", "")
    metadata["userid"] = record.get("userid", "")
    return metadata


# Load JSON Documents
def load_and_chunk_documents():
    loader = JSONLoader(file_path=JSON_path,
        jq_schema='.[]',
        text_content=False,
        metadata_func=metadata_func
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} documents from JSON")
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    chunked_docs = text_splitter.split_documents(docs)
    
    print(f"Split into {len(chunked_docs)} chunks")
    return chunked_docs


def generate_embeddings(texts: List[str]):
    """Generate embeddings for multiple texts"""
    vectors = embeddings_model.embed_documents(texts)
    print(f"Generated {len(vectors)} embeddings of dimension {len(vectors[0])}")
    return vectors


# Insert records to BQ
def insert_rows(client: bigquery.Client, rows_to_insert):
    errors = client.insert_rows_json(full_table_id, rows_to_insert)
    if errors:
        print("Error occurred while inserting records to BQ")
        for error in errors:
            print(error)
    else:
        print(f"Inserted {len(rows_to_insert)} rows")


# Add docs to BQ
def add_docs_to_bq(client :bigquery.client, documents):
    contents=[doc.page_content for doc in documents]

    embeddings=generate_embeddings(contents)
    rows_to_insert = []

    for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
        doc_id=str(uuid.uuid4)
        metadata=dict(doc.metadata)
        rows_to_insert.append({
            "id": doc_id,
            "content": doc.page_content,
            "metadata": json.dumps(metadata),
            "embedding": embedding
        })
        if len(rows_to_insert) >= 1000:
            insert_rows(client, rows_to_insert)
            rows_to_insert.clear()
    if rows_to_insert:
        insert_rows(client, rows_to_insert)
    
    print(f"Successfully added {len(documents)} documents to BigQuery")


# Query Similar Docs
def query_similar_docs(client: bigquery.Client, query_text: str, top_k: int =3):
    
    query_vector = embeddings_model.embed_query(query_text)
    vector_str = "[ " + ",".join(map(str, query_vector)) + " ]"

#    print(f"Searching BQ for query --> {query_vector}")
    search_query = f"""
    SELECT base.id,  base.content,  base.metadata,  distance
    FROM VECTOR_SEARCH(
      TABLE `{full_table_id}`, 'embedding',
      (SELECT {vector_str} AS query_vector),
      distance_type => 'COSINE', 
      top_k => {top_k}
    )
    ORDER BY distance ASC
    """
    print("-"*50)
    try:
        results = client.query(search_query).result()
        for row in results:
            print(f"ID: {row.id}")
            print(f"Content: {row.content[:200]}...")  # Show first 200 chars
            print(f"Distance: {row.distance:.4f} (Similarity: {1 - row.distance:.4f})")
            meta = json.loads(row.metadata) if isinstance(row.metadata, str) else row.metadata
            print(f"Metadata: {meta}")
            print("-"*50)
    except Exception as e:
        print(f"Error occured during Vector Search: {e}")


# Query by Metadata
def query_by_metadata(client: bigquery.Client, metadata_filter: dict, top_k: int =3):

    where_conditions=[]
    for key, value in metadata_filter.items():
        where_conditions.append(f"JSON_VALUE(metadata, '$.{key}') = '{value}'")
    where_clause = " AND ".join(where_conditions)

    query = f"""
    SELECT id, content, metadata
    FROM `{full_table_id}` 
    WHERE {where_clause} LIMIT {top_k}
    """
    print(f"Querying for metadata -> {metadata_filter}")
    print(f"Query to run -> {query}")
    print("-"*50)

    try:
        results = client.query(query).result()
        for row in results:
            print(f"ID: {row.id}")
            print(f"Content: {row.content[:200]}...")  # Show first 200 chars
            meta = json.loads(row.metadata) if isinstance(row.metadata, str) else row.metadata
            print(f"Metadata: {meta}")
            print("-"*50)
    except Exception as e:
        print(f"Error occured during Metadata Search: {e}") 


def main():
    client = bigquery.Client(project=project_id)
    # Create table if not exists
    create_table(client)
    # Load and chunk documents
    chunked_docs = load_and_chunk_documents()
    # Add docs to BQ
    add_docs_to_bq(client, chunked_docs)
    # Query DB
    query_similar_docs(client, "Who is Anand", top_k=2)
    # Query by metadata
    query_by_metadata(client, {"name": "Anand"}, top_k=5)


if __name__ == "__main__":
    main()