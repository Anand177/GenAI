import chromadb


# Persistent client -> Embedded (Local) Mode
client=chromadb.PersistentClient("C:/Learning/chromaDB/text") # Local path to store embeddings

# create collection with SBERT embedding model& HNSW for semantic search
# Model will be downloaded to local (One Time activity)
collection = client.get_or_create_collection(name="paraphrase-albert-small-v2",
                                             metadata={"hnsw:space" : "cosine"})
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

ids=[]  # Every item to embedd need an Id (like HBase Id)
metadata=[] # Metadata on the document to embedd

for i in range(len(corpus)):
    ids.append("Id-" + str(i))
    # Add Sentence length, word count, dummy URL as metadata
    metadata.append({"length": len(corpus[i]), 
                     "word_count": len(corpus[i].split()), 
                     "url": "https:link"+str(i)})
    
# Index data
collection.add(documents=corpus,metadatas=metadata,ids=ids)

print(f"Collection Count -> {collection.count()}")
result =collection.peek(4)

# Print data
print(result["ids"])
print(result["documents"])
print(result["embeddings"])
print(result["included"])
print(result["metadatas"])

# Query with custom conditions
result=collection.query(
    query_texts="I love Cooking",
    n_results=3,        # Get 3 nearby result
# Look only for records with word count < 6
# $gt, $gte, $lt, $lte, $ne, $eq, $in, $nin
    where={"word_count" : {"$lte" : 6}},  
# Look only for document with given string
# $contains, $not_contains, $regex, $not_regex, $and, $or
#    where_document={"$contains" : "man"}    
)

# Print data
print(result["ids"])
print(result["documents"])
print(result["metadatas"])


# Get by ID
result = collection.get(ids=['Id-3', 'Id-30'])

print(result["ids"])
print(result["documents"])
print(result["metadatas"])
