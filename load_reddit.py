import pandas as pd
import gzip
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize import TreebankWordTokenizer


def load_reddit():
    # || Load the persona data || #
    def load_persona_data():
        with gzip.open('data/social_chemistry_posts.gzip', 'rb') as f:
            data = pd.read_pickle(f)
        return data

    # Load the persona data
    data = load_persona_data()
    data = data.dropna(subset=['author_fullname']) # Drop rows with missing author_fullname as then we can not create a persona for them

    # # Save the persona data to csv
    # data.to_csv('data/social_chemistry_posts.csv', index=False)


    # Self disclosure phrases
    self_disclosure_phrases = pd.read_csv('data/self_disclosure_phrases.txt', header=None, names=['phrase'])
    self_disclosure_phrases = self_disclosure_phrases['phrase'].tolist()
    # print(self_disclosure_phrases)

    # Search for self disclosure phrases in the posts
    filtered_posts = data[data['fulltext'].str.contains(r'\b(?:' + '|'.join(self_disclosure_phrases) + r')\b', case=False, na=False)]
    # print(filtered_posts['fulltext'])

    # test_sample = filtered_posts.sample(n=3000, random_state=42)
    test_sample = filtered_posts


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
    return filtered_tokenized_sentences


def get_posts_with_authors():
    # || Load the persona data || #
    def load_persona_data():
        with gzip.open('data/social_chemistry_posts.gzip', 'rb') as f:
            data = pd.read_pickle(f)
        return data

    # Load the persona data
    data = load_persona_data()
    data = data.dropna(subset=['author_fullname']) # Drop rows with missing author_fullname as then we can not create a persona for them
    
    # Save the persona data to csv
    # data.to_csv('data/social_chemistry_posts.csv', index=False)


    # Self disclosure phrases
    self_disclosure_phrases = pd.read_csv('data/self_disclosure_phrases.txt', header=None, names=['phrase'])
    self_disclosure_phrases = self_disclosure_phrases['phrase'].tolist()
    # print(self_disclosure_phrases)

    # Search for self disclosure phrases in the posts
    # filtered_posts = data[data['fulltext'].str.contains(r'\b(?:' + '|'.join(self_disclosure_phrases) + r')\b', case=False, na=False)]
    
    # || Tokenize the text || #
    # Create a Punkt tokenizer
    tokenizer = PunktSentenceTokenizer()

    author_sentence_pairs = []
    for _, row in data.iterrows():
        author = row['author_fullname']
        sentences = tokenizer.tokenize(row['fulltext'])
        
        # Filter tokenized sentences with self-disclosure phrases
        for sentence in sentences:
            if any(phrase in sentence for phrase in self_disclosure_phrases):
                author_sentence_pairs.append((author, sentence))

    return author_sentence_pairs

def load_labels():
    tokenizer = PunktSentenceTokenizer()


    def get_csv():
        with open('data/social_comments_filtered.csv', 'r', encoding="utf-8") as f:
            rows = pd.read_csv(f)
        return rows
    

    data = get_csv()
    data = data.dropna(subset=['author_fullname']) # Drop rows with missing author_fullname as then we can not create a training point for them
    data = data.dropna(subset=['body']) # Drop rows with missing body as then we can not create a training point for them
    data = data.dropna(subset=['label']) # Drop rows with missing label as then we can not create a training point for them

    author_sentence_pairs = []
    for _, row in data.iterrows():
        author = row['author_fullname']
        sentences = tokenizer.tokenize(row['body'])

    for sentence in sentences:
        author_sentence_pairs.append((author, sentence))

    return data, author_sentence_pairs

