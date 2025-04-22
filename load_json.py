# Load data from joan in json format
import os
import pandas as pd
from nltk.tokenize.punkt import PunktSentenceTokenizer

def separate_lists(l):
    new_list = []
    
    for author, text in l:
        if len(text) > 1:
            new_list.extend((author, t) for t in text)  # Expands multi-character strings
        elif len(text) == 1:
            new_list.append((author, text[0]))  # Adds single-character items unchanged
    
    return new_list

def load_json(folder='data/aita_filtered_history'):
    # load every json file in the folder
        
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