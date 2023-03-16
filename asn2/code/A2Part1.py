from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm
from math import sqrt
import json

# Function to read tweets from tweets-utf-8.json file
def get_tweets():
    with open('tweets-utf-8.json') as f:
        tweets = [json.loads(line)['text'] for line in f]
    return tweets

def sort_by_sim(query_embedding,document_embeddings,documents):
    similarityList = []
    normalizedQuery = query_embedding / norm(query_embedding)
    for index in range(len(documents)):
        normalizedDocument = document_embeddings[index] / norm(document_embeddings[index])
        denominator = dot(normalizedDocument,normalizedQuery)
        # print(denominator)
        if denominator == 0:
            similarity = 0
        else:
            similarity = dot(query_embedding,document_embeddings[index]) / denominator
        similarityList.append((similarity,documents[index]))
    return sorted(similarityList,key= lambda a: a[0],reverse=True)
    
def glove_top25(query,documents):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')    
    return sort_by_sim(model.encode(query),model.encode(documents),documents)[:25]

def minilm_top25(query,documents):
    model = SentenceTransformer('sentence-transformers/average_word_embeddings_glove.840B.300d')
    return sort_by_sim(model.encode(query),model.encode(documents),documents)[:25]
        
## Test Code

tweets = get_tweets()

print("**************GLOVE*****************")
for p in glove_top25("I am looking for a job.",tweets): print(p)

print("**************MINILM*****************")
for p in minilm_top25("I am looking for a job.",tweets): print(p)