
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
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering

from minisom import MiniSom


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

test_sample = filtered_posts.sample(n=3000, random_state=42)

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


# || Visualize || #
print("Visualizing Clusters")
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(sentence_embeddings)

# Print variance of each PCA dimension
print("PCA Variance Ratios:")
print(pca.explained_variance_ratio_)

# || Cluster the Sentences || #
# Cluster with kMeans
# print("Clustering Sentences with kMeans")
# # Number of clusters
# cluster_count = 3

# kmeans = KMeans(n_clusters=cluster_count, random_state=42)
# labels = kmeans.fit_predict(sentence_embeddings)
# centroids = kmeans.cluster_centers_


# # Find the representative sentences for each cluster
# for cluster_num in range(3):  # Assuming 3 clusters
#     print(f"\nCluster {cluster_num} Representative Sentences:")
#     cluster_indices = np.where(labels == cluster_num)[0]
    
#     # Calculate distances to cluster centroid
#     distances = [np.linalg.norm(sentence_embeddings[idx] - centroids[cluster_num]) for idx in cluster_indices]
#     sorted_indices = cluster_indices[np.argsort(distances)[:5]]  # Top 5 closest sentences
    
#     for idx in sorted_indices:
#         print(tokenized_sentences[idx])   # Print the original sentence


# # Plot the kmeans clusters with pca
# plt.figure(figsize=(10, 10))
# sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=labels, palette='viridis')
# plt.title("Clustered Sentences")
# plt.xlabel("PCA Dimension 1")
# plt.ylabel("PCA Dimension 2")
# plt.savefig('kmeans_cluster_pca.png')

# #Cluster with kNN
# print("Clustering Sentences with kNN")

# # #Number of neighbors
# n_neighbors = 3
# knn = NearestNeighbors(n_neighbors=n_neighbors,algorithm='auto', metric='cosine')
# knn.fit(sentence_embeddings)

# distances, indices = knn.kneighbors(sentence_embeddings)

# DBSCAN
# db_pca = PCA(n_components=50, random_state=42)
# db_reduced_embeddings = db_pca.fit_transform(sentence_embeddings)
print("Clustering Sentences with DBSCAN")
dbscan = DBSCAN(eps=1, min_samples=3, metric='cosine')
dbscan_labels = dbscan.fit_predict(sentence_embeddings)


# Plot the dbscan clusters with pca
plt.figure(figsize=(10, 10))
sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=dbscan_labels, palette='viridis')
plt.title("Clustered Sentences")
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.savefig('dbscan_cluster_pca.png')

# # Agglomerative Clustering
# print("Clustering Sentences with Agglomerative Clustering")
# agg = AgglomerativeClustering(n_clusters=cluster_count)
# agg_labels = agg.fit_predict(sentence_embeddings)

# # Plot the agglomerative clusters with pca
# plt.figure(figsize=(10, 10))
# sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=agg_labels, palette='viridis')
# plt.title("Clustered Sentences")
# plt.xlabel("PCA Dimension 1")
# plt.ylabel("PCA Dimension 2")
# plt.savefig('Agglomerative_cluster_pca.png')

# Self organizing maps
print("Clustering Sentences with Self Organizing Maps")
som = MiniSom(x=10, y=10, input_len=len(sentence_embeddings[0]), sigma=1.0, learning_rate=0.5)
som.train_random(sentence_embeddings, 100)
som_labels = np.array([som.winner(x) for x in sentence_embeddings])

# Extract x and y coordinates
x_coords = [label[0] for label in som_labels]
y_coords = [label[1] for label in som_labels]

# Plot the self organizing maps clusters
plt.figure(figsize=(10, 10))
sns.scatterplot(x=x_coords, y=y_coords, c='blue', alpha=0.7)
plt.title("Clustered Sentences")
plt.xlabel("SOM X Coordinate")
plt.ylabel("SOM Y Coordinate")
plt.grid()
plt.savefig('SOM_cluster_pca.png')

# Spectral Clustering
print("Clustering Sentences with Spectral Clustering")
spectral = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=42)
spectral_labels = spectral.fit_predict(sentence_embeddings)

# Plot the spectral clusters with pca
plt.figure(figsize=(10, 10))
sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=spectral_labels, palette='viridis')
plt.title("Clustered Sentences")
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.savefig('spectral_cluster_pca.png')

# || Visualize || #
print("Visualizing Clusters")
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(sentence_embeddings)

# Print variance of each PCA dimension
print("PCA Variance Ratios:")
print(pca.explained_variance_ratio_)

# # Apply t-SNE to reduce the embeddings to 2 dimensions
# tsne = TSNE(n_components=2, random_state=42)
# tsne_embeddings = tsne.fit_transform(sentence_embeddings)

# # Print variance for the two t-SNE components
# variance_tsne = np.var(tsne_embeddings, axis=0)
# print("Variance of t-SNE components:")
# print(variance_tsne)









# # Plot the kmeans clusters with t-SNE
# plt.figure(figsize=(10, 10))
# sns.scatterplot(x=tsne_embeddings[:, 0], y=tsne_embeddings[:, 1], hue=labels, palette='viridis')
# plt.title("Clustered Sentences")
# plt.xlabel("PCA Dimension 1")
# plt.ylabel("PCA Dimension 2")
# plt.savefig('kmeans_cluster_tsne.png')

# # Plot kNN clusters
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
# plt.savefig('knn_cluster_pca.png')

