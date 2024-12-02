
# Plan: 
# Use NLTK Punkt Tokenizer to tokenize the text
# Use SBERT to generate sentence embeddings and then start clustering 
# Cluster with kNN first 

import gzip
import time
import pickle
import numpy as np
import pandas as pd
import pyLDAvis
import matplotlib.pyplot as plt
import seaborn as sns


# Import the necessary libraries
import nltk
from nltk.tokenize import PunktSentenceTokenizer, word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA


# Download punkt
nltk.download('punkt')

# || Load the persona data || #
def load_persona_data():
    with gzip.open('data/social_chemistry_posts.gzip', 'rb') as f:
        data = pd.read_pickle(f)
    return data

# Load the persona data
data = load_persona_data()
data = data.dropna(subset=['author_fullname']) # Drop rows with missing author_fullname as then we can not create a persona for them
print(data)

# Save the persona data to csv
# data.to_csv('data/social_chemistry_posts.csv', index=False)


# Self disclosure phrases
self_disclosure_phrases = pd.read_csv('data/self_disclosure_phrases.txt', header=None, names=['phrase'])
self_disclosure_phrases = self_disclosure_phrases['phrase'].tolist()
print(self_disclosure_phrases)

# Search for self disclosure phrases in the posts
filtered_posts = data[data['fulltext'].str.contains(r'\b(' +'|'.join(self_disclosure_phrases)+ r')\b', case=False, na=False)]
filtered_posts['fulltext'] = filtered_posts['fulltext'].str.lower() # Make posts all lower case 
print(filtered_posts['fulltext'])

test_sample = filtered_posts.sample(n=10000, random_state=42)

print("Punkt Tokenizer")
# || Tokenize the text || #
# Create a Punkt tokenizer
tokenizer = PunktSentenceTokenizer()

tokenized_sentences = []
for post in test_sample['fulltext']:
    tokenized_sentences.extend(tokenizer.tokenize(post))

# print("Tokenized Sentences:")
# print(tokenized_sentences)

# || Generate Embeddings With SBERT || #
# Load the SBERT model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate the sentence embeddings
sentence_embeddings = sbert_model.encode(tokenized_sentences)

print("Generated Sentence Embeddings Shape:")
print(sentence_embeddings.shape)

# || Cluster the Sentences || #
# Cluster with kMeans
print("Clustering Sentences with kMeans")

# Number of clusters
cluster_count = 3

kmeans = KMeans(n_clusters=cluster_count, random_state=42)
labels = kmeans.fit_predict(sentence_embeddings)

#Cluster with kNN
print("Clustering Sentences with kNN")

#Number of neighbors
n_neighbors = 3
knn = NearestNeighbors(n_neighbors=n_neighbors,algorithm='auto', metric='cosine')
knn.fit(sentence_embeddings)

distances, indices = knn.kneighbors(sentence_embeddings)


# || Visualize || #
print("Visualizing Clusters")
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(sentence_embeddings)

# Plot the kmeans clusters
plt.figure(figsize=(10, 10))
sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=labels, palette='viridis')
plt.title("Clustered Sentences")
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.show()
# plt.figure(figsize=(10, 10))
# for cluster in range(cluster_count):
#     cluster_points = reduced_embeddings[np.array(labels) == cluster]
#     plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster}")
# plt.legend()
# plt.title("Clustered Sentences")
# plt.xlabel("PCA Dimension 1")
# plt.ylabel("PCA Dimension 2")

# # Plot the kNN clusters
# plt.figure(figsize=(10, 10))
# for i, embedding in enumerate(reduced_embeddings):
#     plt.scatter(embedding[0], embedding[1])
#     for neighbor_index in indices[i]:
#         neighbor_embedding = reduced_embeddings[neighbor_index]
#         plt.plot(
#             [embedding[0], neighbor_embedding[0]],
#             [embedding[1], neighbor_embedding[1]],
#             'k--', alpha=0.5
#         )

# plt.title("kNN Relationships (PCA)")
# plt.xlabel("PCA Dimension 1")
# plt.ylabel("PCA Dimension 2")
# plt.show()