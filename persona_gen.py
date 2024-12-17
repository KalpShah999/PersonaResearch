import gzip
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

from itertools import islice


# Load the persona data
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
print(filtered_posts)

# Split the data by author
authors = filtered_posts.groupby('author_fullname')

# print results
print('Number of authors:', len(authors))
print('Number of posts:', len(filtered_posts))


# Calculating posts per author and authors with multiple posts
# Count the number of posts for each author
author_post_count = filtered_posts.groupby('author_fullname').size()

# Group authors by the amount of posts they have
authors_grouped_by_post_count = author_post_count.groupby(author_post_count)

# Count the number of authors with each post count
author_count_by_post_count = authors_grouped_by_post_count.size()

# Filter authors who have more than 1 post
authors_with_multiple_posts = author_post_count[author_post_count > 1]

# Get the number of authors with more than 1 post
num_authors_with_multiple_posts = authors_with_multiple_posts.count()

# Group authors by the amount of posts they have 
big_authors_grouped_by_post_count = authors_with_multiple_posts.groupby(authors_with_multiple_posts)

# Count the number of authors with each post count
big_author_count_by_post_count = big_authors_grouped_by_post_count.size()




# Calculating self disclosure phrases per author
# Count the number of self disclosure phrases for each author
author_phrase_count = authors['fulltext'].apply(lambda x: x.str.count(r'\b(' + '|'.join(self_disclosure_phrases) + r')\b', flags=re.IGNORECASE).sum())

# Group authors by the amount of self disclosure phrases they have
authors_grouped_by_phrase_count = author_phrase_count.groupby(author_phrase_count)

# Count the number of authors with each phrase count
author_count_by_phrase_count = authors_grouped_by_phrase_count.size()


# Calculate the most common self disclosure phrases
# Count the number of times each self disclosure phrase appears
phrase_count = filtered_posts['fulltext'].str.extractall(r'\b(' + '|'.join(self_disclosure_phrases) + r')\b', flags=re.IGNORECASE).groupby(0).size()

# Add any phrases that were not found to the phrase count
for phrase in self_disclosure_phrases:
    if phrase.lower() not in phrase_count:
        phrase_count[phrase] = 0





# Find information that comes after self disclosure phrases
def find_phrase_index(sentence, phrases):
    # For each self-disclosure phrase, find its index in the sentence
    indices = {}
    for phrase in phrases:
        match = re.search(r'\b' + re.escape(phrase) + r'\b', sentence, flags=re.IGNORECASE)
        if match:
            indices[phrase] = match.start()
    return indices

# Isolate sentences that contain self disclosure phrases
sentences = filtered_posts['fulltext'].str.split(r'[.!?]').explode()
# Get sentences that contain self disclosure phrases
sentences_with_phrases = sentences[sentences.str.contains(r'\b(' + '|'.join(self_disclosure_phrases) + r')\b', flags=re.IGNORECASE, na=False)]
# Get the text which follows the self disclosure phrases
text_after_phrase = sentences_with_phrases.apply(
    lambda sentence: {
        phrase: sentence[index+len(phrase):].strip() 
        for phrase, index in find_phrase_index(sentence, self_disclosure_phrases).items()
    }
)
sentences_after_phrases = text_after_phrase.apply(lambda x: list(x.values())[0])
print("Sentences after phrases:")
print(sentences_after_phrases)


### Unigram ###
# Find the most common words that are after self disclosure phrases
# Get the words that come after self disclosure phrases
words_after_phrase = sentences_after_phrases.str.split().explode()
# Count the number of times each word appears
after_word_count = words_after_phrase.value_counts()
# Order the words by frequency
after_word_count = after_word_count.sort_values(ascending=False)
# Print the most common words
print(after_word_count)


### Bigram ###
# Find the most common bigrams that are after self disclosure phrases
bigrams_after_phrase = sentences_after_phrases.str.split().apply(lambda x: list(zip(x, x[1:]))).explode()
# Count the number of times each bigram appears
after_bigram_count = bigrams_after_phrase.value_counts()
# Order the bigrams by frequency
after_bigram_count = after_bigram_count.sort_values(ascending=False)
# Print the most common bigrams
print(after_bigram_count)

### Trigram ###
# Find the most common trigrams that are after self disclosure phrases
trigrams_after_phrase = sentences_after_phrases.str.split().apply(lambda x: list(zip(x, x[1:], x[2:]))).explode()
# Count the number of times each trigram appears
after_trigram_count = trigrams_after_phrase.value_counts()
# Order the trigrams by frequency
after_trigram_count = after_trigram_count.sort_values(ascending=False)
# Print the most common trigrams
print(after_trigram_count)


# Find information that comes before self disclosure phrases
# Get the text which follows the self disclosure phrases
text_before_phrase = sentences_with_phrases.apply(
    lambda sentence: {
        phrase: sentence[:index].strip()
        for phrase, index in find_phrase_index(sentence, self_disclosure_phrases).items()
    }
)

sentences_before_phrases = text_before_phrase.apply(lambda x: list(x.values())[0]).str.strip()
print("Sentences before phrases:")
print(sentences_before_phrases)

### Unigram ###
# Find the most common words that are before self disclosure phrases
words_before_phrase = sentences_before_phrases.str.split().explode()
# Count the number of times each word appears
before_word_count = words_before_phrase.value_counts()
# Order the words by frequency
before_word_count = before_word_count.sort_values(ascending=False)
# Print the most common words
print(before_word_count)

### Bigram ###
# Find the most common bigrams that are before self disclosure phrases
bigrams_before_phrase = sentences_before_phrases.str.split().apply(lambda x: list(zip(x, x[1:]))).explode()
# Count the number of times each bigram appears
before_bigram_count = bigrams_before_phrase.value_counts()
# Order the bigrams by frequency
before_bigram_count = before_bigram_count.sort_values(ascending=False)
# Print the most common bigrams
print(before_bigram_count)

### Trigram ###
# Find the most common trigrams that are before self disclosure phrases
trigrams_before_phrase = sentences_before_phrases.str.split().apply(lambda x: list(zip(x, x[1:], x[2:]))).explode()
# Count the number of times each trigram appears
before_trigram_count = trigrams_before_phrase.value_counts()
# Order the trigrams by frequency
before_trigram_count = before_trigram_count.sort_values(ascending=False)
# Print the most common trigrams
print(before_trigram_count)



# Print examples of sentences divided into before, self disclosure phrase, and after
# Print the first 10 sentences
print('Examples of sentences divided into before, self disclosure phrase, and after:')
for sentence in islice(sentences_with_phrases, 10):
    for phrase, index in find_phrase_index(sentence, self_disclosure_phrases).items():
        phrase_length = len(phrase)
        before = sentence[:index]
        after = sentence[index+phrase_length:]
        print('Full sentence:', sentence)
        print('Before:', before)
        print('Phrase:', phrase.strip())
        print('After:', after)
        print()


# || --------- Generate plots --------- || #

# # Plot the number of posts per author among all authors
# plt.figure()
# author_count_by_post_count.plot(kind='bar')
# plt.title('Number of posts per author with self disclosure phrases')
# plt.xlabel('Author')
# plt.ylabel('Number of posts')
# plt.savefig('num_post_per_author.png')

# # Plot the number of posts per author among authors with more than 1 post
# plt.figure()
# big_author_count_by_post_count.plot(kind='bar')
# plt.title('Number of posts per author with self disclosure phrases')
# plt.xlabel('Author')
# plt.ylabel('Number of posts')
# plt.savefig('num_post_per_author_more_than_1.png')

# # Plot the number of self disclosure phrases per author
# plt.figure()
# author_count_by_phrase_count.plot(kind='bar')
# plt.title('Number of self disclosure phrases per author')
# plt.xlabel('Number of self disclosure phrases')
# plt.ylabel('Number of Authors')
# plt.savefig('num_self_disclosure_phrases_per_author.png')


# # Sorted bar chart of number of matches per self disclosure phrase
# plt.figure()
# phrase_count_sorted = phrase_count.sort_values(ascending=False)
# phrase_count_sorted.plot(kind='bar', figsize=(10, 6))
# plt.xticks(rotation=45, ha='right')

# # Add the number of matches to the top of each bar
# for i, value in enumerate(phrase_count_sorted.values): 
#     plt.text(i, value + 100, str(value), ha='center', va='bottom', fontsize=9)

# plt.title('Matches per self disclosure phrase', fontsize=16)
# plt.xlabel('Self disclosure phrase', fontsize=14)
# plt.ylabel('Number of matches', fontsize=14)
# plt.tight_layout()
# plt.savefig('matches_per_self_disclosure_phrase.png')


# # Most common phrases after self disclosure phrases (to end of sentence)
# # plot top 20 most common unigrams after self disclosure phrases
# plt.figure()
# after_word_count.head(20).plot(kind='bar', figsize=(10, 6))
# plt.xticks(rotation=45, ha='right')
# plt.title('Most common words after self disclosure phrases', fontsize=16)
# plt.xlabel('Word', fontsize=14)
# plt.ylabel('Frequency', fontsize=14)
# plt.tight_layout()
# plt.savefig('common_word_after.png')


# # plot 20 most common digrams after self disclosure phrases
# plt.figure()
# after_bigram_count.head(20).plot(kind='bar', figsize=(10, 6))
# plt.xticks(rotation=45, ha='right')
# plt.title('Most common bigrams after self disclosure phrases', fontsize=16)
# plt.xlabel('Bigram', fontsize=14)
# plt.ylabel('Frequency', fontsize=14)
# plt.tight_layout()
# plt.savefig('common_two_after.png')

# # plot 20 most common trigrams after self disclosure phrases
# plt.figure()
# after_trigram_count.head(20).plot(kind='bar', figsize=(10, 6))
# plt.xticks(rotation=45, ha='right')
# plt.title('Most common trigrams after self disclosure phrases', fontsize=16)
# plt.xlabel('Trigram', fontsize=14)
# plt.ylabel('Frequency', fontsize=14)
# plt.tight_layout()
# plt.savefig('common_three_after.png')


# # Most common phrases before self disclosure phrases (from start of sentence)
# # plot top 20 most common unigrams before self disclosure phrases
# plt.figure()
# before_word_count.head(20).plot(kind='bar', figsize=(10, 6))
# plt.xticks(rotation=45, ha='right')
# plt.title('Most common words before self disclosure phrases', fontsize=16)
# plt.xlabel('Word', fontsize=14)
# plt.ylabel('Frequency', fontsize=14)
# plt.tight_layout()
# plt.savefig('common_word_before.png')

# # plot 20 most common digrams before self disclosure phrases
# plt.figure()
# before_bigram_count.head(20).plot(kind='bar', figsize=(10, 6))
# plt.xticks(rotation=45, ha='right')
# plt.title('Most common bigrams before self disclosure phrases', fontsize=16)
# plt.xlabel('Bigram', fontsize=14)
# plt.ylabel('Frequency', fontsize=14)
# plt.tight_layout()
# plt.savefig('common_two_before.png')

# # plot 20 most common trigrams before self disclosure phrases
# plt.figure()
# before_trigram_count.head(20).plot(kind='bar', figsize=(10, 6))
# plt.xticks(rotation=45, ha='right')
# plt.title('Most common trigrams before self disclosure phrases', fontsize=16)
# plt.xlabel('Trigram', fontsize=14)
# plt.ylabel('Frequency', fontsize=14)
# plt.tight_layout()
# plt.savefig('common_three_before.png')



