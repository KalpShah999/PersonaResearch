# Use the LIWC dictionary to count the number of words in a text that fall into a given category. 
# Then group into categories and plot the results.
# Categories:
# - Demographic (identity)
# - Expierences (Work/occupation, Education, Hobbies, Habits)
# - Attitude/opinion about something (values, beliefs, mental state)
# - Relations to others(social, family, friends)

import re
import matplotlib.pyplot as plt
import load_reddit
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from colored import Fore, Back, Style
from collections import defaultdict

demographic_categories = ['male', 'female', 'netspeak', 'pronoun']
experiences_categories = ['percept', 'bio', 'work', 'home', 'money', 'relig', 'death']
attitude_categories = ['discrep', 'achievement', 'power', 'reward', 'risk', 'affect']
relations_categories = ['family', 'friend', 'affiliation']


def load_liwc_dict(filename):
    liwc_dict = defaultdict(lambda: [])
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: # skip empty lines
                continue

            word, category = line.split(',')
            word = word.strip()
            category = category.strip()
            
            # Store mapping 
            if word in liwc_dict:
                liwc_dict[word].append(category)
            else: 
                liwc_dict[word] = [category]
    return liwc_dict

def count_categories(text, liwc_dict):
    # Count the number of words in a text that fall into a given category
    # Return a dictionary with the counts for each category

    words  = text.lower().split()
    liwc = []
    print_this = 0
    print_string = ""
    counts = {}       

    print(
            f"demographic: {Fore.GREEN}green{Style.RESET}, "
            f"experiences: {Fore.YELLOW}yellow{Style.RESET}, "
            f"attitude: {Fore.MAGENTA}magenta{Style.RESET}, "
            f"relations: {Fore.CYAN}cyan{Style.RESET}"
        )    
    print(words)
    for word in words:
        if word in liwc_dict:
            categories = liwc_dict[word]
            for category in categories:
                if category in counts:
                    counts[category] += 1
                else:
                    counts[category] = 1

        else:
            for key in liwc_dict:
                liwc.append(key)
                if key.endswith('*'): # check for wildcard
                    base_word = key[:-1]
                    if word.startswith(base_word):
                        categories = liwc_dict[key]
                        for category in categories:
                            if category in counts:
                                counts[category] += 1
                            else:
                                counts[category] = 1
        
        printed = False  
        if liwc_dict[word] == []:
            print_string += (f'{Style.reset}' + word + ' ')     
        else:
            for category in liwc_dict[word]:
                category = category.strip().lower()
                # if category in demographic_categories:
                #     print_string += (f'{Fore.GREEN}' + word + ' ')
                #     print_this = True
                #     printed = True
                #     print(f"Word: {Fore.GREEN}{word}{Style.reset}")
                #     print(f"Cateogry: {Fore.GREEN}{category}{Style.reset}")
                #     print("-------------------------------------------------")
                # if category in experiences_categories:
                #     print_string += (f'{Fore.yellow}' + word + ' ')
                #     print_this = True
                #     printed = True
                #     print(f"Word: {Fore.yellow}{word}{Style.reset}")
                #     print(f"Cateogry: {Fore.yellow}{category}{Style.reset}")
                #     print("-------------------------------------------------")
                # if category in attitude_categories:
                #     print_string += (f'{Fore.magenta}' + word + ' ')
                #     print_this = True
                #     printed = True
                #     print(f"Word: {Fore.magenta}{word}{Style.reset}")
                #     print(f"Cateogry: {Fore.magenta}{category}{Style.reset}")
                #     print("-------------------------------------------------")
                if category in relations_categories:
                    print_string += (f'{Fore.cyan}' + word + ' ')
                    print_this = True
                    printed = True
                    print(f"Word: {Fore.cyan}{word}{Style.reset}")
                    print(f"Cateogry: {Fore.cyan}{category}{Style.reset}")
                    print("-------------------------------------------------")
                # If the category is the final category in liwc_dict[word] and has not been printed then print it
                if category.upper() == liwc_dict[word][-1] and not printed:
                    print_string += (f'{Style.reset}' + word + ' ')
                    printed = True
            

        # if "COGPROC" in liwc_dict[word]:
        #     print_this = True
        #     print_string += (f'{Fore.red}' + word + ' ')
        # else:
        #     print_string += (f'{Style.reset}' + word + ' ')
    
    if print_this:
        print("Sentence: " + print_string)
        print("-------------------------------------------------")
        input()

            
    return counts

def group_categories(counts):
    # group the categories into broader categories
    # Return a dictionary with the counts for each group
    # - Demographic (identity)
    # - Expierences (Work/occupation, Education, Hobbies, Habits)
    # - Attitude/opinion about something (values, beliefs, mental state)
    # - Relations to others(social, family, friends)

    group_counts = {'demographic': 0, 'experiences': 0, 'attitude': 0, 'relations': 0}

    for category, count in counts.items():
        category = category.strip().lower()
        if category in demographic_categories:
            group_counts['demographic'] += count
        elif category in experiences_categories:
            group_counts['experiences'] += count
        elif category in attitude_categories:
            group_counts['attitude'] += count
        elif category in relations_categories:
            group_counts['relations'] += count

    return group_counts

def graph_categories(group_counts):
    # Graph the counts for each group
    labels = list(group_counts.keys())
    values = list(group_counts.values())

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.title('LIWC Category Distribution in Texts')
    plt.show()

def cluster_texts(data, num_clusters=3):
    X = np.array(data)  # Convert list of feature vectors into a NumPy array

    # Standardize features for better clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    return cluster_labels


def main():
    liwc_dict = load_liwc_dict('data\LIWC.2015.all')
    posts = load_reddit.load_reddit()

    all_counts = []
    for text in posts:
        counts = count_categories(text, liwc_dict)
        group_counts = group_categories(counts)
        all_counts.append(list(group_counts.values()))
    
    df = pd.DataFrame(all_counts, columns=['demographic', 'experiences', 'attitude', 'relations'])
    # print(df.head())
    # Plot category distributions (overall)
    # avg_counts = df.mean().to_dict()  # Compute mean counts for visualization
    # graph_categories(avg_counts)

    # # Cluster posts using KMeans
    # cluster_labels = cluster_texts(all_counts, num_clusters=4)

    # # count the number of posts in each cluster
    # cluster_counts = pd.Series(cluster_labels).value_counts()
    # print(cluster_counts)

    # # Add cluster labels to DataFrame
    # df['Cluster'] = cluster_labels
    # # print(df.head())

    # # Print cluster assignment for each post
    # for i, (text, cluster) in enumerate(zip(posts, cluster_labels)):
    #     print(f"Post {i+1}: Cluster {cluster}\nText: {text}\n")


    # graph clusters of scatter plot with PCA
    # pca = PCA(n_components=2)
    # X_pca = pca.fit_transform(df.drop(columns='Cluster'))

    # plt.figure(figsize=(10, 5))
    # for cluster in range(4):
    #     cluster_mask = df['Cluster'] == cluster
    #     plt.scatter(X_pca[cluster_mask, 0], X_pca[cluster_mask, 1], label=f'Cluster {cluster}', alpha=0.5)
    # plt.xlabel('Principal Component 1')
    # plt.ylabel('Principal Component 2')
    # plt.title('KMeans Clustering of Posts')
    # plt.legend()
    # plt.show()


main()