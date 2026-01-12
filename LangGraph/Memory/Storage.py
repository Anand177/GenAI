from langgraph.store.memory import InMemoryStore

mem_store=InMemoryStore()

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

sessions=mem_store.search(mem_namespace)
print(sessions)

# Filter by specific criteria
sessions = mem_store.search(mem_namespace,
    filter={"when": "Dec2024"}  # exact match filter
)
print("Filtered results:")
print(sessions)