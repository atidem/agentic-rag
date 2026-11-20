
from sentence_transformers import SentenceTransformer 

##############################################################

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "The model makes up facts that are not true.",          # 0
    "Language models sometimes hallucinate information.",   # 1  same meaning, other words
    "The agent calls a search tool to find information.",   # 2  related topic
    "I had pasta for dinner yesterday.",                    # 3  unrelated
]

vectors = model.encode(sentences)
print(vectors.shape)

scores = model.similarity(vectors[0:1],vectors)
print(scores)
