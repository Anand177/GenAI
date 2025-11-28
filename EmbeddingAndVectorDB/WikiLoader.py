from langchain_community.document_loaders import WikipediaLoader

wiki_loader=WikipediaLoader("Gemini (language model)", lang="en", load_max_docs=5)
wiki_docs = wiki_loader.load()

print(len(wiki_docs))

print(f"Metadata -> {wiki_docs[0].metadata}")
print(f"COntent -> {wiki_docs[0].page_content}")