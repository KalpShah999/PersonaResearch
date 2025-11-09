"""Coreference resolution analysis for text using Stanza NLP.

This module performs coreference resolution to identify pronouns and their
antecedents in text, calculating resolution ratios for sentences. It uses
the Stanza NLP library for coreference chain extraction and NLTK for 
part-of-speech tagging.
"""

import stanza
import nltk
import pandas as pd
import gzip
from collections import defaultdict
from nltk.tokenize import PunktSentenceTokenizer, TreebankWordTokenizer
from nltk.corpus import wordnet


stanza.download('en')


class degree:
    """Store the degree of resolved and unresolved references in a sentence.
    
    Attributes
    ----------
    degree_of_resolve : int
        The number of resolved coreferences in a sentence.
    degree_of_unresolve : int
        The number of unresolved coreferences in a sentence.
    """
    
    def __init__(self, degree_of_resolve, degree_of_unresolve):
        """Initialize the degree object.
        
        Parameters
        ----------
        degree_of_resolve : int
            The number of resolved coreferences.
        degree_of_unresolve : int
            The number of unresolved coreferences.
        """
        self.degree_of_resolve = degree_of_resolve
        self.degree_of_unresolve = degree_of_unresolve


def get_wordnet_pos(tag):
    """Convert NLTK POS tags to WordNet POS tags.
    
    Parameters
    ----------
    tag : str
        An NLTK part-of-speech tag.
    
    Returns
    -------
    str
        The corresponding WordNet POS tag (ADJ, VERB, NOUN, or ADV).
    """
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

# # || Load the persona data || #
# def load_persona_data():
#     with gzip.open('data/social_chemistry_posts.gzip', 'rb') as f:
#         data = pd.read_pickle(f)
#     return data

# # Load the persona data
# data = load_persona_data()
# data = data.dropna(subset=['author_fullname']) # Drop rows with missing author_fullname as then we can not create a persona for them
# print(data)

# # Save the persona data to csv
# # data.to_csv('data/social_chemistry_posts.csv', index=False)


# # Self disclosure phrases
# self_disclosure_phrases = pd.read_csv('data/self_disclosure_phrases.txt', header=None, names=['phrase'])
# self_disclosure_phrases = self_disclosure_phrases['phrase'].tolist()
# print(self_disclosure_phrases)

# # Search for self disclosure phrases in the posts
# filtered_posts = data[data['fulltext'].str.contains(r'\b(' +'|'.join(self_disclosure_phrases)+ r')\b', case=False, na=False)]
# print(filtered_posts['fulltext'])

# # check if there is a post with text 'I then told him how I'
# # print("Checking filter")
# # pd.set_option('display.max_colwidth', None)
# # print(filtered_posts[filtered_posts['fulltext'].str.contains('I then told him how I')]['fulltext'])
# # test = filtered_posts[filtered_posts['fulltext'].str.contains('I then told him how I')]['fulltext']
# # # print the matched self disclosure phrases
# # print(test.str.extractall(r'\b(' +'|'.join(self_disclosure_phrases)+ r')\b'))

# test_sample = filtered_posts.sample(n=1, random_state=42)

print("Tokenize, lemmatize, and POS tag the text")




nlp = stanza.Pipeline(lang='en', processors='tokenize,ner,pos,coref')
# nlp = stanza.Pipeline(lang='en', processors='tokenize,lemma,pos,ner')



# print()

# Get the coreference clusters
# corefchains = doc.coref
# resolved_count = 0
# unresolved_count = 0


def get_coref_chains(doc):
    """Extract coreference chains from a Stanza document.
    
    Parameters
    ----------
    doc : stanza.Document
        A processed Stanza document containing coreference information.
    
    Returns
    -------
    tuple
        A tuple of (cluster_ids, cluster_mentions) where cluster_ids is a list
        of unique identifiers and cluster_mentions is a list of mention objects.
        Returns (None, None) if no coreference chains are found.
    """
    if doc.coref:
        cluster_ids = []
        cluster_mentions = []
        for idx, cluster in enumerate(doc.coref):
            # Generate a unique unit ID for each cluster
            cluster_id = f"unit-id{idx}"
            cluster_ids.append(cluster_id)
            print(f"{cluster_id}:")
            mentions = []
            for mention in cluster.mentions:
                # Each mention has a start and end token index
                start, end = mention.start_word, mention.end_word
                # Extract the mention text from the document
                mention_text = " ".join([word.text for word in doc.sentences[mention.sentence].words[start:end]])
                print(f"  Mention: {mention_text}, Start: {start}, End: {end}, Sentence: {mention.sentence}")
                mentions.append(mention)
            cluster_mentions.append(mentions)
        return cluster_ids, cluster_mentions

    else:
        # print("No coreference chains found.")
        return None, None

# texts = ["I met Lady Gaga", "We met Lady Gaga", "She met Lady Gaga", "They met Lady Gaga", "My brother met Lady Gaga", "She met her"]
# texts = ["John is a guy. He likes cheese. He has a dog. The dog is cute."]
# for text in texts:
#     doc1 = nlp(text)
#     print("{:C}".format(doc1))
    # print all the coreference chains
    # get_coref_chains(doc1)
    # coref_chain1 = doc1.coref
    # # print(coref_chain1)
    # for coref in doc1.coref:
    #     # print(coref)
    #     for mention in coref.mentions:
    #         # print(mention)
    #         mention_text = " ".join([word.text for word in doc1.sentences[mention.sentence].words[mention.start_word:mention.end_word]])
    #         print(f"  Coref Index: {coref.index}, Representative text: {coref.representative_text }, Representative index: {coref.representative_index }")
    #         print(f"  Mention: {mention_text}, Start: {mention.start_word}, End: {mention.end_word}, Sentence: {mention.sentence}")
    # print("=====================================")


    
def print_coref_info(cluster_id, mentions):
    """Print coreference chain information.
    
    Parameters
    ----------
    cluster_id : list
        List of cluster identifiers.
    mentions : list
        List of mention objects corresponding to each cluster.
    """
    for cluster_id, mentions in zip(cluster_id, mentions):
        print(f"{cluster_id}:")
        for mention in mentions:
            start, end = mention.start_word, mention.end_word
            mention_text = " ".join([word.text for word in doc.sentences[mention.sentence].words[start:end]])
            print(f"  Mention: {mention_text}, Start: {start}, End: {end}, Sentence: {mention.sentence}")

def resolve_ratio(degree_of_resolve, degree_of_unresolve):
    """Calculate the resolution ratio for a sentence.
    
    Parameters
    ----------
    degree_of_resolve : int
        The number of resolved coreferences.
    degree_of_unresolve : int
        The number of unresolved coreferences.
    
    Returns
    -------
    float
        The resolution ratio (1.0 if all references are resolved).
    """
    if degree_of_unresolve == 0:
        return 1
    else:
        return degree_of_resolve / (degree_of_resolve + degree_of_unresolve)


def calculate_resolve_ratio(degree_dict):
    """Calculate resolution ratios for all sentences in a dictionary.
    
    Parameters
    ----------
    degree_dict : dict
        Dictionary mapping sentences to degree objects.
    
    Returns
    -------
    dict
        Dictionary mapping sentences to their resolution ratios.
    """
    ratio_dict = {}
    for key, value in degree_dict.items():
        ratio = resolve_ratio(value.degree_of_resolve, value.degree_of_unresolve)
        ratio_dict[key] = ratio
    return ratio_dict



tokenizer = PunktSentenceTokenizer()
word_tokenizer = TreebankWordTokenizer()
tokenized_sentences = []
lemmatized_sentences = []
pos_tagged_sentences = []
# for post in test_sample['fulltext']:
# coref on post
# doc = nlp(post)
# post = "John Bauer works at Stanford, he likes it.  He has been there 4 years. She hates it. Bill loves it."
post = "I met Lady Gaga. We met Lady Gaga. She met Lady Gaga. They met Lady Gaga. My brother met Lady Gaga. She met her. Katy like cans."
doc = nlp(post)
print("{:C}".format(doc))
cluster_ids, cluster_mentions = get_coref_chains(doc)
print_coref_info(cluster_ids, cluster_mentions)

# tokenize
tokenized = tokenizer.tokenize(post)
tokenized_sentences.extend(tokenized)

# lists of resolved and unresolved sentences
unresolved_sentences = []
resolved_sentences = []
degree_dict = defaultdict(degree)





for i, sentence in enumerate(doc.sentences):
    print("=====================================")
    print("Processing sentence:")

    words = [word.text for word in sentence.words]  # Extract words efficiently
    print(words)
    
    # POS tagging for the words
    pos_tags = nltk.pos_tag(words)
    # print("pos_tags")
    # print(pos_tags)

    # Degrees of resolve and unresolve
    degree_of_resolve = 0
    degree_of_unresolve = 0


    sentence_cluster_mentions = [
        (cluster_id, mention)
        for cluster_id, cluster in zip(cluster_ids, cluster_mentions)  # Pair cluster_id with mentions
        for mention in cluster  # Iterate over mentions in each cluster
        if mention.sentence == i  # Filter mentions that belong to the current sentence
    ]

    # print("sentence_cluster_mentions")
    # for mentions in sentence_cluster_mentions:
    #     id = mentions[0]
    #     mention = mentions[1]
    #     print(id)
    #     print(mention)
    #     print(" ".join([word.text for word in doc.sentences[mention.sentence].words[mention.start_word :mention.end_word]]))
    # print(sentence_cluster_mentions)

    if sentence_cluster_mentions:
        mention_counts = defaultdict(int)
        for cluster_id, mention in sentence_cluster_mentions:
            mention_counts[cluster_id] += 1
        unresolved_indexes = [key for key, value in mention_counts.items() if value == 1]
        unresolved_mentions = [mention for cluster_id, mention in sentence_cluster_mentions if cluster_id in unresolved_indexes]
        # print("unresolved_mentions")
        # print(unresolved_mentions)

        if unresolved_mentions:
            for mention in unresolved_mentions:
                start, end = mention.start_word, mention.end_word
                mention_text = " ".join(words[start:end])
                pos = nltk.pos_tag([mention_text])[0][1]  # POS tag the mention
                print("mention_text: ", mention_text)
                print("pos: ", pos)

                # Check if it's a pronoun or adverb (like "there" in "he works there")
                if pos in {'PRP', 'PRP$', 'RB'}:
                    print("unresolved word: ", mention_text)
                    if degree_of_unresolve == 0:
                        unresolved_sentences.append(" ".join(words))
                    degree_of_unresolve += 1


                else:
                    print("First else: "," ".join(words))
                    print("pos: ", pos)
                    degree_of_resolve += 1

                if degree_of_unresolve == 0:
                    resolved_sentences.append(" ".join(words))
        else: 
            print("Middle else: "," ".join(words))
            resolved_sentences.append(" ".join(words))
            degree_of_resolve += 1
    else:
        print("Final else: "," ".join(words))
        resolved_sentences.append(" ".join(words))
        degree_of_resolve += 1
    
    # Add the degree of resolve and unresolve to the dictionary
    degree_dict[sentence.text] = degree(degree_of_resolve, degree_of_unresolve)

# Print counts
print(f"Resolved sentences count: {len(resolved_sentences)}")
print(f"Unresolved sentences count: {len(unresolved_sentences)}")

# # print resolved and unresolved sentences
# print("Resolved sentences:")
# for sentence in resolved_sentences:
#     print(sentence)

# print("Unresolved sentences:")
# for sentence in unresolved_sentences:
#     print(sentence)

for key, value in degree_dict.items():
    print(f"Sentence: {key}, Degree of resolve: {value.degree_of_resolve}, Degree of unresolve: {value.degree_of_unresolve}")

# Calculate resolve ratio
ratio_dict = calculate_resolve_ratio(degree_dict)
ratio_dict = dict(sorted(ratio_dict.items(), key=lambda item: item[1], reverse=True)) # sort the dictionary by value in descending order
print("Resolve ratio:")
for key, value in ratio_dict.items():
    print(f"Sentence: {key}, Resolve ratio: {value}")








# if cluster_mentions:
#             unresolved_mentions = [mentions for mentions in cluster_mentions if len(mentions) == 1]
#             unresolved_ids = [cluster_id for cluster_id, mentions in zip(cluster_ids, cluster_mentions) if len(mentions) == 1]
#             if unresolved_mentions:
#                 for cluster_id, mentions in zip(unresolved_ids, unresolved_mentions):
#                     for unresolved_mention in mentions:
#                         start, end = unresolved_mention.start_word, unresolved_mention.end_word
#                         mention_text = " ".join([word.text for word in doc.sentences[unresolved_mention.sentence].words[start:end]])
#                         # Get the POS tag for the unresolved mention
#                         pos = nltk.pos_tag([mention_text])[0][1]
#                         word = mention_text
#                         if pos in ['PRP', 'PRP$']:
#                             # print("Unresolved mention:")
#                             # print(f"Word: {word}, POS: {pos}")
#                             unresolved_sentences.append(sentence)
#                         else:
#                             resolved_sentences.append(sentence)
#     print("Resolved sentences count:")
#     print(len(resolved_sentences))
#     print("Unresolved sentences count:")
#     print(len(unresolved_sentences))





# # Iterate over sentences and tokens
# for sentence in doc.sentences:
#     print(f"Processing sentence: {sentence.text}")
#     for word in sentence.words:
#         print(f"Word: {word.text}, Lemma: {word.lemma}, POS: {word.upos}")
#     # Named entity recognition
#     print("\nNamed Entities:")
#     for entity in sentence.ents:
#         print(f"Entity: {entity.text}, Type: {entity.type}")




# stanza.download('en')
# nlp = stanza.Pipeline("en", processors="tokenize,coref")
# # nlp = stanza.Pipeline(lang='en', processors='tokenize,lemma,pos,ner')


# doc = nlp("Barack Obama was born in Hawaii. He was elected president in 2008.")

# print("{:C}".format(doc))

# # Get the coreference clusters
# clusters = doc.coref
# resolved_count = 0
# unresolved_count = 0

# # Iterate over clusters
# resolved_mentions = set()
# if clusters:
#     for cluster in clusters:
#         for mention in cluster.mentions:
#             resolved_mentions.add((mention.start_word, mention.end_word))
#             resolved_count += 1
            
    

# # All mentions in the document
# all_mentions = set()
# for sentence in doc.sentences:
#     for token in sentence.tokens:
#         all_mentions.add((token.start_word, token.end_word))

# # Unresolved mentions are the ones not in resolved_mentions
# unresolved_mentions = all_mentions - resolved_mentions

# # Count resolved and unresolved mentions
# resolved_count = len(resolved_mentions)
# unresolved_count = len(unresolved_mentions)

# # Print the mentions
# print("Resolved Mentions:")
# for start, end in resolved_mentions:
#     print(" ".join([word.text for word in doc.sentences[0].words[start:end]]))

# print("\nUnresolved Mentions:")
# for start, end in unresolved_mentions:
#     print(" ".join([word.text for word in doc.sentences[0].words[start:end]]))

# print(f"Resolved mentions: {resolved_count}")
# print(f"Unresolved mentions: {unresolved_count}")