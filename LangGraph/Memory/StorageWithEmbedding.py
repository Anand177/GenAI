from dotenv import load_dotenv
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langgraph.store.memory import InMemoryStore

import os

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001",
                google_api_key=GOOGLE_API_KEY,
                task_type="semantic_similarity")
# Other available tasks -> retrieval_document, retrieval_query, classification, 

mem_store=InMemoryStore(index={
    "embed": embeddings,
    "dims": 768 # embedding-001 has 768 dimensions
})

mem_namespace=("memories", "Anand")
mem_key="profile"
mem={"name" : "Anand", "residence" : "Chennai", "hobby" : ["photography", "kick-boxing"]}

mem_store.put(mem_namespace, mem_key, mem)
anand_profile=mem_store.get(namespace=mem_namespace, key=mem_key)
print(anand_profile)

anand_profile=mem_store.search(mem_namespace)
print(anand_profile)


mem_namespace=("memories", "Anand", "PlacesVisited")
mem_store.put(mem_namespace, "111", value={"id":1, "place": "Kodaikanal", "when" : "Sept2025"})
mem_store.put(mem_namespace, "222", value={"id":2, "place": "Thekkady", "when" : "Mar2024"})
mem_store.put(mem_namespace, "333", value={"id":3, "place": "Anamalai", "when" : "Dec2024"})
mem_store.put(mem_namespace, "AAA", value={"id":1, "place": "Pondicherry", "when" : "Apr2025"})

sessions=mem_store.search(mem_namespace, query="Mountains", limit=5)
print("Semantic Search")
print(sessions)

# Filter by specific criteria
