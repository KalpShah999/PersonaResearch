
# Plan: 
# Use NLTK Punkt Tokenizer to tokenize the text
# Use SBERT to generate sentence embeddings and then start clustering 
# Cluster with kNN first 

import gzip
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re


# Import the necessary libraries
import nltk
from nltk.tokenize import PunktSentenceTokenizer, TreebankWordTokenizer
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering


# from minisom import MiniSom


# Download punkt
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')

# Function to convert NLTK POS tags to WordNet POS tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

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
print(filtered_posts['fulltext'])

# check if there is a post with text 'I then told him how I'
# print("Checking filter")
# pd.set_option('display.max_colwidth', None)
# print(filtered_posts[filtered_posts['fulltext'].str.contains('I then told him how I')]['fulltext'])
# test = filtered_posts[filtered_posts['fulltext'].str.contains('I then told him how I')]['fulltext']
# # print the matched self disclosure phrases
# print(test.str.extractall(r'\b(' +'|'.join(self_disclosure_phrases)+ r')\b'))

# test_sample = filtered_posts.sample(n=3000, random_state=42)
test_sample = filtered_posts

print("Tokenize, lemmatize, and POS tag the text")
# || Tokenize the text || #
# Create a Punkt tokenizer
tokenizer = PunktSentenceTokenizer()
word_tokenizer = TreebankWordTokenizer()
lemmatizer = nltk.WordNetLemmatizer()

tokenized_sentences = []
lemmatized_sentences = []
pos_tagged_sentences = []
for post in test_sample['fulltext']:
    # tokenize
    tokenized = tokenizer.tokenize(post)
    tokenized_sentences.extend(tokenized)
    # Tokenize sentences into words, then lemmatize each word

    # for sentence in tokenized:
    #     words = word_tokenizer.tokenize(sentence)  # Word tokenization
        
    #     # POS tagging for the words
    #     pos_tags = nltk.pos_tag(words)
    #     # print("pos_tags")
    #     # print(pos_tags)
        
    #     # Lemmatize each word (optionally use POS tag for better lemmatization)
    #     lemmatized = [
    #         lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag))
    #         for word, tag in pos_tags
    #     ]
    #     lemmatized_sentences.append(lemmatized)
        
    #     # Append POS tagging for the sentence
    #     pos_tagged_sentences.append(pos_tags)


# Filter tokenized sentences with self disclosure phrases
filtered_tokenized_sentences = []
for sentence in tokenized_sentences:
    if any(phrase in sentence for phrase in self_disclosure_phrases):
        filtered_tokenized_sentences.append(sentence)

print(filtered_tokenized_sentences)

# print("Tokenized Sentences:")
# print(tokenized_sentences)
# print("POS Tagged Sentences:")
# print(pos_tagged_sentences[10])

# # || Visualize PoS Tags || #
# # Bar graph which shows the frequency of each PoS tag
# pos_tags = [tag[1] for sentence in pos_tagged_sentences for tag in sentence]
# pos_counts = pd.Series(pos_tags).value_counts()
# plt.figure(figsize=(10, 10))
# sns.barplot(x=pos_counts.values, y=pos_counts.index)
# plt.title("PoS Tag Distribution")
# plt.xlabel("Count")
# plt.ylabel("PoS Tag")
# plt.savefig('PoS_tags.png')

# # graph the distribution of verbs 
# verbs = [tag[0] for sentence in pos_tagged_sentences for tag in sentence if tag[1].startswith('V')]
# verb_counts = pd.Series(verbs).value_counts()
# plt.figure(figsize=(10, 10))
# sns.barplot(x=verb_counts.values[:60], y=verb_counts.index[:60])
# plt.title("Most Common Verbs")
# plt.xlabel("Count")
# plt.ylabel("Verb")
# plt.savefig('verbs.png')
# print("Most Common Verbs:")
# print(verb_counts[:60])




# || Generate Embeddings With SBERT || #
# Load the SBERT model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate the sentence embeddings
sentence_embeddings = sbert_model.encode(filtered_tokenized_sentences, show_progress_bar=True)

print("Generated Sentence Embeddings Shape:")
print(sentence_embeddings.shape)


# || Visualize || #
cluster_count = 10

print("Visualizing Clusters")
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(sentence_embeddings)

# Print variance of each PCA dimension
print("PCA Variance Ratios:")
print(pca.explained_variance_ratio_)

# || Cluster the Sentences || #
# Cluster with kMeans
print("Clustering Sentences with kMeans")
# Number of clusters

kmeans = KMeans(n_clusters=cluster_count, random_state=42)
labels = kmeans.fit_predict(sentence_embeddings)
centroids = kmeans.cluster_centers_


# Find the representative sentences for each cluster
for cluster_num in range(cluster_count):
    print(f"\nCluster {cluster_num} Representative Sentences:")
    cluster_indices = np.where(labels == cluster_num)[0]
    
    # Calculate distances to cluster centroid
    distances = [np.linalg.norm(sentence_embeddings[idx] - centroids[cluster_num]) for idx in cluster_indices]
    sorted_indices = cluster_indices[np.argsort(distances)[:5]]  # Top 5 closest sentences
    
    for idx in sorted_indices:
        print(filtered_tokenized_sentences[idx])   # Print the original sentence

# print a random sample of 5 from each cluster
for cluster_num in range(cluster_count):
    print(f"\nCluster {cluster_num} Random Sample:")
    cluster_indices = np.where(labels == cluster_num)[0]
    random_sample = np.random.choice(cluster_indices, 5)
    for idx in random_sample:
        print(filtered_tokenized_sentences[idx])

unique_clusters = np.unique(labels)
print(f"Number of unique clusters: {len(unique_clusters)}")

# print the number of sentences in each cluster
cluster_counts = pd.Series(labels).value_counts()
print(cluster_counts)

# Plot the kmeans clusters with pca
plt.figure(figsize=(10, 10))
scatter = sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=labels, palette='tab10')
plt.title("Clustered Sentences")
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
handles, _ = scatter.get_legend_handles_labels()
plt.legend(handles, ["Family Dynamics/Relationships", "Stress, Anxiety, and Depression", "Friendship and Social Interactions", "Conflict and Confrontation", 
                     "Communications with males", "Education and Employment", "Obligations and Compromising", "Self-Reflection on morality", 
                     "Avoiding communication", "Romantic Relationships with females"], title="Cluster Labels")
plt.savefig('kmeans_cluster_pca_labeled.png')

# #Cluster with kNN
# print("Clustering Sentences with kNN")

# # #Number of neighbors
# n_neighbors = 3
# knn = NearestNeighbors(n_neighbors=n_neighbors,algorithm='auto', metric='cosine')
# knn.fit(sentence_embeddings)

# distances, indices = knn.kneighbors(sentence_embeddings)

# # DBSCAN
# # db_pca = PCA(n_components=50, random_state=42)
# # db_reduced_embeddings = db_pca.fit_transform(sentence_embeddings)
# print("Clustering Sentences with DBSCAN")
# dbscan = DBSCAN(eps=1, min_samples=3, metric='cosine')
# dbscan_labels = dbscan.fit_predict(sentence_embeddings)


# # Plot the dbscan clusters with pca
# plt.figure(figsize=(10, 10))
# sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=dbscan_labels, palette='viridis')
# plt.title("Clustered Sentences")
# plt.xlabel("PCA Dimension 1")
# plt.ylabel("PCA Dimension 2")
# plt.savefig('dbscan_cluster_pca.png')

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

# # Self organizing maps
# print("Clustering Sentences with Self Organizing Maps")
# som = MiniSom(x=10, y=10, input_len=len(sentence_embeddings[0]), sigma=1.0, learning_rate=0.5)
# som.train_random(sentence_embeddings, 100)
# som_labels = np.array([som.winner(x) for x in sentence_embeddings])

# # Extract x and y coordinates
# x_coords = [label[0] for label in som_labels]
# y_coords = [label[1] for label in som_labels]

# # Plot the self organizing maps clusters
# plt.figure(figsize=(10, 10))
# sns.scatterplot(x=x_coords, y=y_coords, c='blue', alpha=0.7)
# plt.title("Clustered Sentences")
# plt.xlabel("SOM X Coordinate")
# plt.ylabel("SOM Y Coordinate")
# plt.grid()
# plt.savefig('SOM_cluster_pca.png')

# # Spectral Clustering
# print("Clustering Sentences with Spectral Clustering")
# spectral = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=42)
# spectral_labels = spectral.fit_predict(sentence_embeddings)

# # Plot the spectral clusters with pca
# plt.figure(figsize=(10, 10))
# sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=spectral_labels, palette='viridis')
# plt.title("Clustered Sentences")
# plt.xlabel("PCA Dimension 1")
# plt.ylabel("PCA Dimension 2")
# plt.savefig('spectral_cluster_pca.png')

# # || Visualize || #
# print("Visualizing Clusters")
# pca = PCA(n_components=2)
# reduced_embeddings = pca.fit_transform(sentence_embeddings)

# # Print variance of each PCA dimension
# print("PCA Variance Ratios:")
# print(pca.explained_variance_ratio_)

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

