# Refer SBERT.ipynb
from sentence_transformers import SentenceTransformer, util


#https://sbert.net/docs/sentence_transformer/pretrained_models.html
#model_name = "all-MiniLM-L12-v2"     # Best but large model
model_name = "nreimers/albert-small-v2" # Smallest model    Best coe cos calculation
model = SentenceTransformer(model_name)
#

# Embed words
words = ["cat", "dog", "horse"]
embeddings = model.encode(words)
print(embeddings)

len(embeddings[0])  # Embedding Dimension

test_word = "kitten"
test_embedding = model.encode(test_word)

cos_result = []
euc_result = []
man_result = []
dot_result = []

# Calculate with different measurement
for i, embedding in enumerate(embeddings):
  cos_score = util.cos_sim(test_embedding, embedding)
  cos_result.append((cos_score, words[i]))

  euc_score = util.euclidean_sim(test_embedding, embedding)
  euc_result.append((euc_score, words[i]))

  man_score=util.manhattan_sim(test_embedding, embedding)
  man_result.append((man_score, words[i]))

  dot_score=util.dot_score(test_embedding, embedding)
  dot_result.append((dot_score, words[i]))

cos_result.sort(reverse=True)
print(cos_result)

euc_result.sort(reverse=True)
print(euc_result)

man_result.sort(reverse=True)
print(man_result)

dot_result.sort(reverse=True)
print(dot_result)
