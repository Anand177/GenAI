# Use VisualizeEmbedding.ipynb
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

model = SentenceTransformer("all-MiniLM-L6-v2")
words = ["lemon" , "banana", "guava", "tiger", "elephant", "leopard", "car", "truck", "bicycle",
         "vaccum cleaner", "Automatic machine that cleans the floor without presence of humans"]
embeddings = model.encode(words)
print(f"Original Dimension: {embeddings.shape} ")

pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)
print(f"Reduced Dimensions: {reduced_embeddings.shape}")

for word, embedding in zip(words, embeddings):
    print(f"{word} = ({embedding[0]:7.3f},{embedding[1]:7.3f})")