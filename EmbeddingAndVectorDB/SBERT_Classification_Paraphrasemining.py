# Refer SBERT_Classification_Paraphrasemining.ipynb

from sentence_transformers import SentenceTransformer, util


#**Create Sentence Transformer Model**
model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

#**Classification**
#1. Use 1+ group of similar sentences
#2. Calculate embedding score of all sentence in group, get average
#3. Get Embedding of input sentence, derive cosine score against all sentences in all groups
#4. Calculate average score for each group
#5. Sentence aligns best with group with higher average

# Category : Sports
sports_facts = [
    "Football, also known as soccer, is the most popular sport in the world, with billions of fans worldwide.",
    "Basketball was invented in 1891 by Dr. James Naismith, a Canadian physical education instructor, as an indoor game to keep his students active during the winter months.",
    "Tennis is a highly competitive sport that originated in the 19th century and is played by millions of people around the world on various surfaces such as grass, clay, and hardcourt.",
    "Golf is a precision club-and-ball sport in which players use various clubs to hit balls into a series of holes on a course in as few strokes as possible."
]

# Category : History
history_facts = [
    "The Renaissance was a period of cultural rebirth that emerged in Europe during the 14th to 17th centuries, marking a transition from the Middle Ages to modernity.",
    "The Industrial Revolution, which began in Britain in the late 18th century, transformed society by introducing mechanized manufacturing processes and urbanization.",
    "The Cold War, spanning from the late 1940s to the early 1990s, was a geopolitical conflict between the United States and the Soviet Union, characterized by ideological, economic, and military competition.",
    "The French Revolution, which erupted in 1789, was a watershed moment in European history, leading to the overthrow of the monarchy and the rise of democratic principles."
]

sports_facts_embeddings = model.encode(sports_facts, convert_to_tensor=True)
history_facts_embeddings = model.encode(history_facts, convert_to_tensor=True)

print("Dimension = ", len(sports_facts_embeddings[0]))


#**Derive Cosine Score**

'''def calculate_cos_score(test_embedding, embeddings):
  scores = []
  for i, embedding in enumerate(embeddings):
    score = util.cos_sim(test_embedding, embedding)
    scores.append((score,i))

  avg_score = sum(score for score, index in scores )/len(scores)
  max_score, max_index = max(scores, key=lambda x: x[0])
  return scores, avg_score, max_score, max_index
'''
def  calculate_cosine_score(test_embedding, embeddings):
    # variable holds the score
    scores = []

    # calculate the average score, max score
    average_score = 0

    # holds the info on max score
    max_score = 0
    max_score_index = 0

    # loop through the list to calculate the score, average, max
    for i, embedding in enumerate(embeddings):
        score = util.cos_sim(test_embedding, embedding)
        scores.append(score)
        average_score += score
        if score > max_score:
            max_score_index = i
            max_score = score

    return scores, (average_score)/len(scores), max_score_index


#**Test Sentence Embedding**
test_sentences = [
    "I like putting",
    "two strong armies came face to face",
    "hoops on the two ends of the court",
    "steam engine changed the world",
    "arts, and self expression was the highlight"
]

test_sentence = test_sentences[4]

test_sentence_embedding =  model.encode(test_sentence)

# Calculate the scores
scores_sports, average_score_sports, max_score_index_sports = calculate_cosine_score(test_sentence_embedding, sports_facts_embeddings)
scores_history, average_score_history, max_score_index_history = calculate_cosine_score(test_sentence_embedding, history_facts_embeddings)

### Check category
category = "sports"
if average_score_history > average_score_sports:
    category = "history"

print(test_sentence," - Belongs to category : ", category)

#**ParaPhrase Mining**
#1. For given array of sentences, compare all sentences against each other
#2. Returns list of pairs with highest similarity

input = [
    "A man is eating food.",
    "A man is eating a piece of bread.",
    "A man is riding a horse.",
    "A woman is playing violin.",
    "A man is riding a white horse on an enclosed ground.",
]

mining_result=util.paraphrase_mining(model, input)
type(mining_result)

for result in mining_result:
  print("="*50)
  print (result)
  print(f"Score -> {result[0]}")
  print(f"Sentence 1 -> {input[result[1]]}")
  print(f"Sentence 2 -> {input[result[2]]}")
  print("="*50)
