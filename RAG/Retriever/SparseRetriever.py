#pip install rank_bm25
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

corpus = [
    "Rice is a cereal and staple food for over half the world",
    "It originated in Asia and Africa",
    "India, China and Indonesia are largest consumers of rice",
    "Oryza Sativa is cultivated in India and other parts of Asia",
    "India and China are the largest producers of Rice",
    "Jasmine is a sticky rice variety used in Japan",
    "Indian Basmati is long grain aromatic rice"
]

# Indexing as Text
print("")
print("Corpus Text")
corpus_collected=list(map(str.lower, corpus))
bm25_retriever = BM25Retriever.from_texts(corpus_collected)
bm25_retriever.k=3

result = bm25_retriever.invoke("India Rice")
print(result)

result = bm25_retriever.invoke("Jasmine Rice")
print(result)

# Indexing as Document
print("")
print("Corpus Document")
corpus_docs=[]
for i, dat in enumerate(corpus):
    document=Document(
        page_content=dat,
        metadata= {"doc_num" : i, "source": "Rice Detail # "+str(i)}
    )
    corpus_docs.append(document)

#print(corpus_docs)

bm25_retriever = BM25Retriever.from_documents(documents=corpus_docs, k=3)
#bm25_retriever.k=3

result = bm25_retriever.invoke("India Rice")
print(result)
result = bm25_retriever.invoke("Jasmine Rice")
print(result)
