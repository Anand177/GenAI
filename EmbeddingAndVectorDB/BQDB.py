from dotenv import load_dotenv
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import List

import json
import os
import time

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="C:/Learning/AI/Key/gen-lang-client-0398817262-d752d424f736.json"

project_id="gen-lang-client-0398817262"
dataset_id="GenAi"
table_id="text_embeddings"

full_table_id=f"{project_id}.{dataset_id}.{table_id}"
VECTOR_MODEL = "text-embedding-004" # Google's embedding model for BQ

corpus=[
  "A man is eating food.", "A man is eating a piece of bread.",
  "The chef is preparing a delicious meal in the kitchen.", "A chef is tossing vegetables in a sizzling pan.",
  "A man is riding a horse.", "A man is riding a white horse on an enclosed ground.",
  "A woman is playing violin.", "A musician is tuning his guitar before the concert.",
  "The girl is carrying a baby.", "The baby is giggling while playing with her toys.",
  "The family is having a picnic under the shady oak tree.", "A group of friends is hiking up the mountain trail.",
  "The mechanic is repairing a broken-down car in the garage.", "The old man is feeding breadcrumbs to the ducks at the pond.",
  "The artist is sketching a beautiful landscape at sunset.", "A man is painting a colorful mural on the city wall.",
  "A team of scientists is conducting experiments in the laboratory.", "A group of students is studying together in the library.",
  "The birds are chirping happily in the morning sun.", "The dog is chasing its tail around the backyard.",
  "A group of children are playing soccer in the park.", "A monkey is playing drums.",
  "A boy is flying a kite in the open field.", "Two men pushed carts through the woods.",
  "A woman is walking her dog along the beach.", "A young girl is reading a book under a shady tree.",
  "The dancer is gracefully performing on stage.", "The farmer is harvesting ripe tomatoes from the vine."
]

ids=[] 
metadata=[] 

for i in range(len(corpus)):
    ids.append(f"Id-{i}")
    metadata.append({"length": len(corpus[i]), 
                      "word_count": len(corpus[i].split()), 
                      "url": f"https:link{i}"})


def create_and_index_table(client : bigquery.Client):
    schema = [
        bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("text", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("metadata", "JSON", mode="REQUIRED"),
        bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED")
    ]

    table=bigquery.Table(full_table_id,schema=schema)

    try:
        client.get_table(table=full_table_id)
        print(f"Table -> {full_table_id} exists.")
    except NotFound:
        print(f"Table -> {full_table_id} not found.")
        table = client.create_table(table=table)
        print(f"Table -> {full_table_id} created successfully.")
    except Exception as e:
        print(f"Exception {e} raised")
        print(f"Full Table Id -> {full_table_id}")


def generate_embeddings(texts: List[str]):
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model=VECTOR_MODEL,
        google_api_key=GOOGLE_API_KEY
    )
    vectors=embeddings_model.embed_documents(texts)
    print(f"Generated {len(vectors)} embeddings of dimension {len(vectors[0])}.")

    return vectors


def generate_query_embedding(text: str):
    """Generate embedding for a single query text"""
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model=VECTOR_MODEL,
        google_api_key=GOOGLE_API_KEY
    )
    # Use embed_query for single text - returns a flat list
    vector = embeddings_model.embed_query(text)
    print(f"Generated query embedding of dimension {len(vector)}.")
    return vector


def insert_vector_into_bq(client : bigquery.Client, vectors: List[List[float]]):

    rows_to_insert = []
    for i in range(len(corpus)):
        rows_to_insert.append({
            "id" : ids[i],
            "text" : corpus[i],
            "metadata" : json.dumps(metadata[i]),
            "embedding" : vectors[i]
        })

        if len(rows_to_insert) > 1000:
            bq_insertion(client, rows_to_insert)
            rows_to_insert.clear()

    if rows_to_insert:
        bq_insertion(client, rows_to_insert)
    time.sleep(2)


def bq_insertion(client : bigquery.Client, rows_to_insert):
    errors = client.insert_rows_json(full_table_id, rows_to_insert)

    if errors:
        print("Error occurred while inserting records to BQ")
        for error in errors:
            print(error)
    else :
        print(f"Successfully inserted {len(rows_to_insert)} embeddings to BQ")


def create_vector_index(client : bigquery.Client):
    """Creating BQ Index for fast searches"""

    count_query = f"SELECT COUNT(*) as count FROM `{full_table_id}`"
    result = client.query(count_query).result()
    row_count = next(result).count
    
    if row_count < 5000:
        print(f"Skipping index creation: Table has {row_count} rows (minimum 5000 required for IVF index)")
        print("VECTOR_SEARCH will work without index for small datasets")
        return

    create_index_query=f"""
    CREATE VECTOR INDEX IF NOT EXISTS embedding_index
    ON `{full_table_id}` (embedding)
    OPTIONS(
      distance_type = 'COSINE',
      index_type = 'IVF'
    )
    """
    try:
        print("Creating vector index...")
        client.query(create_index_query).result()
        print("Vector index created successfully!")
    except Exception as e:
        print(f"Index creation failed: {e}")


def query_by_ids(client: bigquery.Client, ids: List[str]):
    """Get Documents by ID"""
    id_str="', '".join(ids)
    query = f"""
    SELECT id, text, metadata
    FROM `{full_table_id}` WHERE id IN ('{id_str}')
    """

    print("Querying DB by ID")
    print("-"*50)
    results = client.query(query=query).result()
    for row in results:
        meta = json.loads(row.metadata) if isinstance(row.metadata, str) else row.metadata
        print(f"ID: {row.id}")
        print(f"Text: {row.text}")
        print(f"Metadata: {meta}")
        print("-" * 50)


def query_similar_vectors(client : bigquery.Client, query_text: str, top_k : int =3):
    """Query for similar vectors using VECTOR_SEARCH"""

    print(f"Querying Similar Vectors for query -> {query_text}")
    query_vector = generate_query_embedding(query_text)
    vector_str = "[" + ",".join(map(str, query_vector)) + "]" # Fmt vector for search

    search_query = f"""
    SELECT 
      base.id,
      base.text,
      base.metadata,
      distance
    FROM VECTOR_SEARCH(
      TABLE `{full_table_id}`,
      'embedding',
      (SELECT {vector_str} AS query_vector),
      distance_type => 'COSINE',
      top_k => {top_k}
    )
    ORDER BY distance ASC
    """

    print("-" * 50)
    try:
        results = client.query(search_query).result()
        for row in results:
            meta = json.loads(row.metadata) if isinstance(row.metadata, str) else row.metadata
            print(f"ID: {row.id}")
            print(f"Text: {row.text}")
            print(f"Distance: {row.distance:.4f} (Similarity: {1 - row.distance:.4f})")
            print(f"Metadata: {meta}")
            print("-" * 50)
    except Exception as e:
        print(f"VECTOR_SEARCH error: {e}")


def main():

    client = bigquery.Client(project=project_id)
    create_and_index_table(client)

    corpus_vectors=generate_embeddings(corpus)
    insert_vector_into_bq(client, corpus_vectors)
    create_vector_index(client) 

    query_by_ids(client, ['Id-0', 'Id-9'])
    query_similar_vectors(client, "I love Cooking", top_k=5)



if __name__ == "__main__":
    main()