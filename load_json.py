"""Load and process JSON data from AITA (Am I The Asshole) Reddit posts.

This module loads JSON files containing Reddit post data, tokenizes the text
into sentences, and pairs sentences with their authors.
"""

import os
import pandas as pd
from nltk.tokenize.punkt import PunktSentenceTokenizer


def separate_lists(l):
    """Separate a list of (author, text_list) tuples into individual sentences.
    
    Parameters
    ----------
    l : list
        List of tuples where each tuple contains (author, list_of_texts).
    
    Returns
    -------
    list
        List of (author, single_text) tuples where each text is separated.
    """
    new_list = []
    
    for author, text in l:
        if len(text) > 1:
            new_list.extend((author, t) for t in text)  # Expands multi-character strings
        elif len(text) == 1:
            new_list.append((author, text[0]))  # Adds single-character items unchanged
    
    return new_list


def load_json(folder='data/aita_filtered_history'):
    """Load all JSON files from a folder and extract author-sentence pairs.
    
    Parameters
    ----------
    folder : str, optional
        Path to the folder containing JSON files (default is 'data/aita_filtered_history').
    
    Returns
    -------
    list
        List of (author, sentence) tuples extracted from all JSON files.
    """
    author_sentence_verdict = []
    tokenizer = PunktSentenceTokenizer()
    for file in os.listdir(folder):
        print(file)
        if file.endswith('.json'): # change after it works 
            with open(f'{folder}/{file}', 'r') as f:
                data = pd.read_json(f)
                data = data.dropna(subset=['body'])
                try: 
                    data = data.dropna(subset=['name'])
                    authors = data['name']
                except KeyError:
                    data = data.dropna(subset=['author'])
                    authors = data['author']
                sentences = map(tokenizer.tokenize, data['body'])
                pairs = separate_lists(list(zip(authors, sentences)))
                author_sentence_verdict.extend(pairs)

    return author_sentence_verdict