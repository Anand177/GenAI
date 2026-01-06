from dotenv import load_dotenv
from google.cloud import bigquery, storage
from google.cloud.exceptions import NotFound
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Optional

import json, os, uuid

# Load env variables
load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Learning/AI/Key/gen-lang-client-0398817262-d752d424f736.json"

# GCP Configuration
project_id = "gen-lang-client-0398817262"
dataset_id = "GenAi"
table_id = "wiki_child_chunks"
full_table_id = f"{project_id}.{dataset_id}.{table_id}"
VECTOR_MODEL = "text-embedding-004"
bkt_name = "anand_genai"
parent_doc_prefix = "parent_docs/"

urls = [
    "https://en.wikipedia.org/wiki/Large_language_model",
    "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
    "https://en.wikipedia.org/wiki/Gemini_(language_model)",
    "https://en.wikipedia.org/wiki/BERT_(language_model)"
]

class BQVectorStore:
    """BQ based vector store for child chunks"""
    def __init__(self, client: bigquery.Client, full_table_id: str, google_api_key: str):
        self.client=client
        self.full_table_id=full_table_id
        self.embeddings_model = GoogleGenerativeAIEmbeddings(model=VECTOR_MODEL,
                    google_api_key=google_api_key)
        
    def create_table(self):
        schema = [
            bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("content", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("metadata", "JSON", mode="REQUIRED"),
            bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED")
        ]
        table=bigquery.Table(self.full_table_id, schema=schema)
        try:
            self.client.get_table(self.full_table_id)
            print(f"Table -> {self.full_table_id} exists")
        except NotFound:
            print(f"Table -> {self.full_table_id} doesn't exist. Creating...")
            self.client.create_table(table=table)
            print(f"Table -> {self.full_table_id} created")

    def add_documents(self, docs: List[Document]):
        """Add Doc as embeddings to BQ"""
        contents = [ doc.page_content for doc in docs]  # Extract content
        embeddings = self.embeddings_model.embed_documents(contents)    # Generate Embeddings
        print(f"Embeddings generated. length -> {len(embeddings)}")

        rows_to_insert=[]
        for doc, embedding in zip(docs, embeddings):
            doc_id=str(uuid.uuid4())
            metadata=dict(doc.metadata)
            rows_to_insert.append({"id" : doc_id,
                    "content" : doc.page_content,
                    "metadata" : json.dumps(dict(doc.metadata)),
                    "embedding" : embedding
                    })
            if len(rows_to_insert) > 500:
                self._insert_rows(rows_to_insert)
                rows_to_insert.clear()
        if rows_to_insert:
            self._insert_rows(rows_to_insert)

    def _insert_rows(self, rows_to_insert):
        """Insert rows to BQ"""
        errors=self.client.insert_rows_json(self.full_table_id, rows_to_insert)
        if errors:
            print(f"Error inserting to BQ -> {errors}")

    def similarity_search(self, query: str, k: int =3):
        """Search for similar docs"""
        query_vector=self.embeddings_model.embed_query(query)
        vector_str= "[" + ",".join(map(str, query_vector)) + "]"

        bq_query = f"""
        SELECT base.id, base.content, base.metadata, distance
        FROM VECTOR_SEARCH(
          TABLE `{self.full_table_id}`, 'embedding',
          (SELECT {vector_str} AS query_vector),
          distance_type => 'COSINE', top_k => {k}
        ) ORDER BY distance ASC
        """
        try:
            results = self.client.query(bq_query).result()
            documents =[]
            for row in results:
                metadata = json.loads(row.metadata) if isinstance(row.metadata, str) else row.metadata
                metadata["distance"] = row.distance

                doc = Document(page_content=row.content, metadata=metadata)
                documents.append(doc)
                return documents
        except Exception as e:
            print(f"Vector Search failed with error -> {e}")
            return []


class GCSParentDocStore:
    """GCS Storage for parent docs"""
    
    def __init__ (self, bkt_name: str, prefix: str = "parent_docs/"):
        self.storage_client=storage.Client()
        self.bkt_name= bkt_name
        self.prefix=prefix
        self.bkt = self.storage_client.lookup_bucket(bkt_name)
        if self.bkt:
            print(f"Bucket --> {bkt_name} exists")
        else:
            print(f"Bucket --> {bkt_name} doesn't exist")
            try:
                self.bkt = self.storage_client.create_bucket(bkt_name)
                print(f"Bucket --> {bkt_name} created")
            except Exception:
                print("Bucket creation failed")
            

    def add_doc(self, doc_id: str, doc : Document):
        """Add doc to GCS"""
        blob_name=f"{self.prefix}{doc_id}.json"
        blob=self.bkt.blob(blob_name=blob_name)
        doc_data= {"page_content" : doc.page_content, "metadata" : doc.metadata}
        blob.upload_from_string(json.dumps(doc_data),
                    content_type="application/json")
        
    def get_doc(self, doc_id: str) -> Optional[Document]:
        blob_name=f"{self.prefix}{doc_id}.json"
        blob=self.bkt.blob(blob_name=blob_name)
        try:
            doc_data = json.loads(blob.download_as_string())
            return Document(page_content=doc_data["page_content"], 
                    metadata= doc_data["metadata"])
        except Exception as e:
            print(f"Error retrieving document {doc_id}: {e}")
            return None
    
    def get_docs(self, doc_ids: List[str]) -> list[Optional[Document]]:
        return [self.get_doc(doc_id) for doc_id in doc_ids]
    
    def list_all_docs(self) -> List[str]:
        """List all parent docs"""
        blobs=self.bkt.list_blobs(prefix=self.prefix)
        doc_ids=[]
        for blob in blobs:
            doc_id = blob.name.replace(self.prefix, "").replace(".json", "")
            if doc_id:
                doc_ids.append(doc_id)
        return doc_ids


class GCPParentDocumentRetriever:
    """Parent Doc Retreiver using GCP products"""

    def __init__(self, vectorstore: BQVectorStore, docstore: GCSParentDocStore,
        child_splitter: RecursiveCharacterTextSplitter, parent_splitter: RecursiveCharacterTextSplitter):
        self.vectorstore = vectorstore
        self.docstore = docstore
        self.child_splitter = child_splitter
        self.parent_splitter = parent_splitter

    def add_docs(self, docs: List[Document]):
        """Split and add documents to child vector store and parent doc store"""
        parent_docs=self.parent_splitter.split_documents(docs)
        print(f"Parent Docs split to {len(parent_docs)} chunks")
        
        all_child_docs=[]
        for parent_doc in parent_docs:
            parent_id = str(uuid.uuid4())
            self.docstore.add_doc(parent_id, parent_doc)    # Store parent doc in GCS
            child_docs=self.child_splitter.split_documents([parent_doc])    # Split to child docs

            for child_doc in child_docs:
                child_doc.metadata["doc_id"] = parent_id
                child_doc.metadata["source"] = parent_doc.metadata.get("source", "")
            all_child_docs.extend(child_docs)
        self.vectorstore.add_documents(all_child_docs)  # Store child docs in BQ

        print(f"Parent docs: {len(parent_docs)} stored in GCS")
        print(f"Child Docs: {len(all_child_docs)} stored in BQ")

    def invoke(self, query: str, k: int =3):
        """Retreive parent docs based on Child Chunks"""
        child_docs = self.vectorstore.similarity_search(query, k) # Search child chunks
        parent_ids = list(set(      # get parent Id
            doc.metadata.get("doc_id")
            for doc in child_docs
            if doc.metadata.get("doc_id")
        ))

        parent_docs=[]
        for parent_id in parent_ids:
            parent_doc=self.docstore.get_doc(parent_id)
            if parent_doc:
                parent_docs.append(parent_doc)

        return parent_docs

        
def main():
    bq_client=bigquery.Client(project=project_id)
    vector_store= BQVectorStore(bq_client, full_table_id, GOOGLE_API_KEY)   # Create BQ Vector store to store child docs
    vector_store.create_table()

    parent_doc_store=GCSParentDocStore(bkt_name,parent_doc_prefix)  # Create GCS parent doc store

    print("Documents to load from Web")
    for i, url in enumerate(urls):
        print(f"Document {i} -> {url}")
    loader = WebBaseLoader(web_paths=urls)
    docs=loader.load()
    print(f"{len(docs)} documents loaded from Web")

    # Create splitters & retriever
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    parent_splitter= RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=20)
    retriever=GCPParentDocumentRetriever(vectorstore=vector_store, docstore=parent_doc_store,
                child_splitter=child_splitter, parent_splitter=parent_splitter)
    print("Adding docs to retriever")
    retriever.add_docs(docs)

    parent_doc_ids = parent_doc_store.list_all_docs()
    print(f"{len(parent_doc_ids)} documents added successfully")

    # Querying Parent Doc Retriever
    input_qs = [
        "What is RAG",
        "What is LLM",
        "What is Fine Tuning",
        "What is chunking"
    ]
    q=2
    query=input_qs[q]
    print("-"*50)
    print(f"Query to search -> {query}")
    print("-"*50)
    print("Retrieving Child chunks")
    
    child_results=vector_store.similarity_search(query, k=3)
    parent_docs=[]
    uniq_parent_ids = set()

    for i, child_doc in enumerate(child_results):
        print(f"Child chunk {i} Metadata: {child_doc.metadata}")
        print(f"Child chunk {i} Content: {child_doc.page_content[:150]}")
        parent_id = child_doc.metadata.get('doc_id')
        if parent_id and parent_id not in uniq_parent_ids:
            parent_doc = parent_doc_store.get_doc(parent_id)
            if parent_doc:
                print(f"Parent Doc Source: {parent_doc.metadata.get('source')}")
                print(f"Parent Doc Length: {len(parent_doc.page_content)}")
                parent_docs.append(parent_doc)
                uniq_parent_ids.add(parent_id)
    print(f"Total Parent IDs retrieved: {len(uniq_parent_ids)}")
    print(f"Total Parent Docs: {len(parent_docs)}")

    if parent_docs:
        context = "\n".join([doc.page_content for doc in parent_docs])
        template = """Use the below context to answer the query. 
        Add your parametric knowledge to only enhance the answer and don't hallucinate
        Question: {query}
        Context: {context}
        """
        prompt_template = PromptTemplate(template=template,
            input_variables=["query", "context"]
        )
        llm = GoogleGenerativeAI( model="gemini-flash-latest",
                google_api_key=GOOGLE_API_KEY,
                temperature=0.7,
                top_p=0.65)
        chain = prompt_template | llm

        print("Generating Answer")
        print("-"*50)
        response = chain.invoke({"query" : query, "context" : context})
        print(f"Answer : {response}")
        print("-"*50)
    else:
        print("No relevant parent docs found")


if __name__ =="__main__":
    main()