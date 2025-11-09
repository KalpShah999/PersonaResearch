"""LDA topic modeling and visualization for Reddit posts.

This module performs Latent Dirichlet Allocation (LDA) topic modeling on Reddit
posts containing self-disclosure phrases. It creates interactive visualizations
of discovered topics using pyLDAvis.
"""

import gzip
import time
import pickle
import numpy as np
import pandas as pd
import pyLDAvis

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re


def extract_relevant_sentences(post, phrases):
    """Extract sentences from a post that contain self-disclosure phrases.
    
    Parameters
    ----------
    post : str
        The text of a post to search.
    phrases : list
        List of self-disclosure phrases to search for.
    
    Returns
    -------
    str or None
        The first sentence containing a self-disclosure phrase, or None if
        no matching sentence is found.
    """
    # Split the post into sentences using punctuation as delimiters
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\\.|\\?|\\!)\\s', post)
    # Check each sentence for the self-disclosure phrases
    for sentence in sentences:
        if re.search(r'\\b(' + '|'.join(phrases) + r')\\b', sentence, re.IGNORECASE):
            return sentence.strip()
    return None  # Return None if no phrase is found


def load_persona_data():
    """Load persona data from a gzip-compressed pickle file.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame containing Reddit post data from the Social Chemistry dataset.
    """
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


# || Visualise || #
# Create a count vectorizer
vectorizer = CountVectorizer(
    stop_words='english',  # Remove common stopwords
    lowercase=True,       # Ensure case consistency
    token_pattern=r'\b[a-zA-Z]{3,}\b'  # Include only words of length >= 3
)
X = vectorizer.fit_transform(filtered_posts['fulltext'])

# Extract the vocabulary and term frequencies
vocabulary = vectorizer.get_feature_names_out()
term_frequencies = np.array(X.sum(axis=0)).flatten()

# Train a LDA model
lda_model = LatentDirichletAllocation(n_components=10, max_iter=10, learning_method='online', random_state=100, n_jobs=-1)
start_time = time.time()
lda_model.fit(X)
print('Elapsed time: ', time.time() - start_time)

# Prepare the LDA visualization
lda_visual = pyLDAvis.prepare(
    topic_term_dists=lda_model.components_, 
    doc_topic_dists=lda_model.transform(X), 
    doc_lengths=np.array(X.sum(axis=1)).flatten(),
    vocab=vocabulary,
    term_frequency=term_frequencies
)

# Display the LDA visualization
pyLDAvis.display(lda_visual)

# Save the LDA visualization
pyLDAvis.save_html(lda_visual, 'lda_visual_10.html')
