import load_reddit
import load_json
import re
import matplotlib.pyplot as plt
from collections import Counter 
import pandas as pd
import torch
import torch.nn as nn
import os

# Demographic patterns
identity_pattern = re.compile(r"\b(I am|I'm|Im)\s*(a|an)?\s*(.+?)(?=[,.!?]|$)", re.IGNORECASE)
age_pattern = re.compile(
    r"\b(?:I am |I'm |Im )(?P<age1>\d{1,2})(?: years old)?\b"  
    r"|"
    r"\((?P<age2>\d{2})(?P<gender>[A-Za-z]{1,4})\)",  # Ensure the (24M) format is captured correctly
    re.IGNORECASE
)
# agetest_strings = [
#     "I'm 24 years old",
#     "I'm 24",
#     "I am 18 years old",
#     "Im 30 years old",
#     "(22M)",
#     "(19NB)",
#     "(35TF)",
#     "7 years old"
# ]

# for text in agetest_strings:
#     match = age_pattern.search(text)
#     if match:
#         age = match.group("age1") or match.group("age2")  # Get the captured age
#         gender = match.group("gender") if match.group("gender") else "N/A"
#         print(f"Text: {text} → Age: {age}, Gender: {gender}")

gender_pattern = re.compile(
    r"\b(?:I am|I'm|Im)\s*"
    r"(?P<sentence_gender>(?:non[- ]?binary|male|female|man|woman|boy|girl|guy|dude|mother|father|sister|brother|son|daughter|husband|wife|trans(?: man| woman|gender)?))\b"
    r"|"
    r"\((?P<age>\d{1,2})(?!ish)(?P<short_gender>[A-Za-z]{1,4})\)",  # Captures (24M)-style format
    re.IGNORECASE
)

# gendertest_strings = [
#     "I'm male",
#     "I am a transgender woman",
#     "Im non-binary",
#     "I am non binary",
#     "I am nonbinary",
#     "(24M)",
#     "(30NB)",
#     "(19TF)",
#     "24ish",
#     "7ish"
# ]

# for text in gendertest_strings:
#     match = gender_pattern.search(text)
#     if match:
#         sentence_gender = match.group("sentence_gender") if match.group("sentence_gender") else "N/A"
#         age = match.group("age") if match.group("age") else "N/A"
#         short_gender = match.group("short_gender") if match.group("short_gender") else "N/A"
#         print(f"Text: {text} → Gender: {sentence_gender}, Age: {age}, Short Gender: {short_gender}")


# Experience patterns
hobby_pattern = re.compile(
    r"\b(?:I like to|I love to|I often|I usually|I prefer|"
    r"I can't stand|I adore|I'm passionate about|I'm fond of|I'm interested in|"
    r"I'm into|I tend to)\s+"
    r"(?P<preference>.+?)(?=[,.!?]|$)",
    re.IGNORECASE
)
have_pattern = re.compile(
    r"\b(?:I have|I've got|I own|I possess|I hold|I acquired|I received)\s+"
    r"(?P<experience>.+?)(?=[,.!?]|$)",
    re.IGNORECASE
)
work_pattern = re.compile(
    r"\b(?:I work|I'm working|I'm employed|I have a job|I do work|I freelance|"
    r"I have been working|I started working|I used to work|"
    r"I studied | I am studying | I'm studying | Im studying | I have studied)\s+"
    r"(?P<job_details>.+?)(?=[,.!?]|$)",
    re.IGNORECASE
)


# Attitude patterns
attitude_pattern = re.compile(
    r"\b(?:I dislike|I like|I enjoy|I love|I hate|I believe|I think|I feel|I support|I oppose|I stand for|I stand against|"
    r"I'm against|I'm for|I'm pro-|I'm anti-|I value|I don't believe in|"
    r"I consider|I advocate for|I reject|I agree with|I disagree with|"
    r"I am passionate about|I'm critical of|I align with|I side with|"
    r"I view|I respect|I distrust|I question|I doubt|I condemn|I appreciate|"
    r"I prioritize|I favor|I disapprove of|I endorse|I subscribe to|"
    r"I'm skeptical of)\s+"
    r"(?P<opinion>.+?)(?=[,.!?]|$)",
    re.IGNORECASE
)


# Relationship patterns
relationship_pattern = re.compile(
    r"\b(?:I am|I'm|Im|My|Our|We are|I have)(a|an)?\s*"
    r"(?P<relationship>"
    r"mother|father|mom|dad|brother|sister|son|daughter|uncle|aunt|grandfather|grandmother|"
    r"grandpa|grandma|cousin|nephew|niece|husband|wife|boyfriend|girlfriend|partner|spouse|"
    r"best friend|friend|roommate|fiancé|fiancée|in-law|step(?:mother|father|brother|sister|son|daughter)"
    r")\b",
    re.IGNORECASE
)

def write_to_file(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, "w", encoding='utf-8') as f:
        for item in data:
            f.write(str(item) + "\n")

def separate_lists(l):
    new_list = []
    
    for text, author in l:
        if len(text) > 1:
            new_list.extend((t, author) for t in text)  # Expands multi-character strings
        else:
            new_list.append((text[0], author))  # Adds single-character items unchanged
    
    return new_list

def find_overlap(list1, list2):
    # list1 = separate_lists(list1)
    # list2 = separate_lists(list2)
    overlap_list = list(set(list1) & set(list2))
    overlap = len(overlap_list)
    return overlap, overlap_list

def check_overlaps(lists, names):
    # Find overlap in lists
    overlaps = []
    overlap_list = []
    for (i, name) in zip(range(len(lists)), names):
        tmp = lists.copy()
        tmp.pop(i)
        overlap = 0
        over_list = []
        for l in (tmp):
            over, over_list_result = find_overlap(lists[i], l)
            overlap += over
            over_list.append(over_list_result)
            flatten(over_list)
        overlaps.append(overlap)
        overlap_list.append(over_list)
    return overlaps, overlap_list

flatten = lambda z: [x for y in z for x in y]

def get_author_dist(posts):
    author_counts = {}
    for post in posts:
        author = post[1]
        count = len(post[0])
        if author in author_counts:
            author_counts[author] += count
        else:
            author_counts[author] = count
    return author_counts

def graph_dist(counter, title, xlabel, ylabel, filename):
    # graph the distribution of the different patterns per author
    plt.figure()
    plt.bar(counter.keys(), counter.values())
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # save
    plt.savefig(filename)

def main():
    sentences_original = load_reddit.get_posts_with_authors()
    # sentences_joan = load_json.load_json()
    labeled_data, labeled_pairs = load_reddit.load_labels()

    # print(f"Original posts: {sentences_original[0]}")
    # print(f"Labeled posts: {labeled_pairs[0]}")
    # print(f"Joan posts: {sentences_joan[0]}")

    sentences = sentences_original + labeled_pairs # + sentences_joan  # TODO: For testing 

    if not sentences:
        print("No posts found.")
        return


    # extracted patterns from posts
    identityList = []
    agesList = []
    genderList = []
    hobbysList = []
    havesList = []
    worksList = []
    attitudesList = []
    relationshipsList = []


    for post in sentences:
        # Demographic patterns
        identities = [match.group() for match in identity_pattern.finditer(post[1])]
        if identities != []: 
            identityList.append((identities, post[0]))

        ages = [match.group("age1") or match.group("age2") for match in age_pattern.finditer(post[1])]
        if ages != []: 
            agesList.append((ages, post[0]))

        genders = [match.group("sentence_gender") or match.group("short_gender") for match in gender_pattern.finditer(post[1])]
        if genders != []: 
            genderList.append((genders, post[0]))

        # Experience patterns
        hobbys = [match.group() for match in hobby_pattern.finditer(post[1])]
        if hobbys != []: 
            hobbysList.append((hobbys, post[0]))

        haves = [match.group() for match in have_pattern.finditer(post[1])]
        if haves != []: 
            havesList.append((haves, post[0]))

        works = [match.group() for match in work_pattern.finditer(post[1])]
        if works != []: 
            worksList.append((works, post[0]))

        # Attitude patterns
        attitudes = [match.group() for match in attitude_pattern.finditer(post[1])]
        if attitudes != []: 
            attitudesList.append((attitudes, post[0]))

        # Family patterns
        relationships = [match.group() for match in relationship_pattern.finditer(post[1])]
        if relationships != []: 
            relationshipsList.append((relationships, post[0]))

        # print(f"post: {post}")

        # print(f"identities: {identities}")
        # print(f"ages: {ages}")
        # print(f"genders: {genders}")
        # print(f"hobbys: {hobbys}")
        # print(f"haves: {haves}")
        # print(f"works: {works}")
        # print(f"attitudes: {attitudes}")
        # print(f"relationships: {relationships}")
        # print(f"families: {families}")

    # separate lists
    identityList = separate_lists(identityList)
    agesList = separate_lists(agesList)
    genderList = separate_lists(genderList)
    hobbysList = separate_lists(hobbysList)
    havesList = separate_lists(havesList)
    worksList = separate_lists(worksList)
    attitudesList = separate_lists(attitudesList)
    relationshipsList = separate_lists(relationshipsList)


    # Get the distribution of the different patterns per author
    identity_authors = Counter(get_author_dist(identityList).values())
    ages_authors = Counter(get_author_dist(agesList).values())
    gender_authors = Counter(get_author_dist(genderList).values())
    hobbys_authors = Counter(get_author_dist(hobbysList).values())
    haves_authors = Counter(get_author_dist(havesList).values())
    works_authors = Counter(get_author_dist(worksList).values())
    attitudes_authors = Counter(get_author_dist(attitudesList).values())
    relationships_authors = Counter(get_author_dist(relationshipsList).values())

    # [KalpShah999] I'm not sure what these graphs are for or where they're supposed to come from
    # # graph
    # graph_dist(identity_authors, "Distribution of Identities per Author", "Number of Phrases", "Number of Authors", "graphs/regex_dist/identity_authors.png")
    # graph_dist(ages_authors, "Distribution of Ages per Author", "Number of Phrases", "Number of Authors", "graphs/regex_dist/ages_authors.png")
    # graph_dist(gender_authors, "Distribution of Genders per Author", "Number of Phrases", "Number of Authors", "graphs/regex_dist/genders_authors.png")
    # graph_dist(hobbys_authors, "Distribution of hobbys per Author", "Number of Phrases", "Number of Authors", "graphs/regex_dist/hobbys_authors.png")
    # graph_dist(haves_authors, "Distribution of Haves per Author", "Number of Phrases", "Number of Authors", "graphs/regex_dist/haves_authors.png")
    # graph_dist(works_authors, "Distribution of Works per Author", "Number of Phrases", "Number of Authors", "graphs/regex_dist/works_authors.png")
    # graph_dist(attitudes_authors, "Distribution of Attitudes per Author", "Number of Phrases", "Number of Authors", "graphs/regex_dist/attitudes_authors.png")
    # graph_dist(relationships_authors, "Distribution of Relationships per Author", "Number of Phrases", "Number of Authors", "graphs/regex_dist/relationships_authors.png")

    lists = [identityList, agesList, genderList, hobbysList, havesList, worksList, attitudesList, relationshipsList]
    names = ["identity", "ages", "genders", "hobbys", "haves", "works", "attitudes", "relationships"]

    overlaps, overlap_list = check_overlaps(lists, names)

    # remove overlap from identity list
    for o in overlap_list[0]:
        identityList = list(set(identityList) - set(o))

    # remove overlap from haves
    for o in overlap_list[5]:
        havesList = list(set(havesList) - set(o))
    for o in overlap_list[7]:
        havesList = list(set(havesList) - set(o))
    for o in overlap_list[3]:
        attitudesList = list(set(attitudesList) - set(o))

    lists = [identityList, agesList, genderList, hobbysList, havesList, worksList, attitudesList, relationshipsList]


    print("no overlap" if sum(check_overlaps(lists, names)[0]) == 0 else "overlap")
    

    


    




    # write each list to a file 
    print("writing to files")
    write_to_file(identityList, "data/regex/identities.txt")
    write_to_file(agesList, "data/regex/ages.txt")
    write_to_file(genderList, "data/regex/gender.txt")
    write_to_file(hobbysList, "data/regex/hobbys.txt")
    write_to_file(havesList, "data/regex/haves.txt")
    write_to_file(worksList, "data/regex/works.txt")
    write_to_file(attitudesList, "data/regex/attitudes.txt")
    write_to_file(relationshipsList, "data/regex/relationships.txt")
    
    print("files written")

    
    # print stats about the patterns found
    total_sentences = len(sentences)
    print(f"Total sentences: {total_sentences}")

    # Demographic stats
    total_identities = sum(len(nouns) for nouns in identityList)
    # print(agesList)
    total_ages = sum(len(ages) for ages in agesList)
    total_gender = sum(len(genders) for genders in genderList)
    total_demographics = total_identities + total_ages + total_gender

    # Experience stats
    total_hobbys = sum(len(hobbys) for hobbys in hobbysList)
    total_haves = sum(len(haves) for haves in havesList)
    total_works = sum(len(works) for works in worksList)
    total_experiences = total_hobbys + total_haves + total_works

    # Attitude stats
    total_attitudes = sum(len(attitudes) for attitudes in attitudesList)

    # Family stats
    total_relationships = sum(len(relationships) for relationships in relationshipsList)

    print(f"Total identities: {total_identities}")
    print(f"Total ages: {total_ages}")
    print(f"Total gender: {total_gender}")
    print(f"Total hobbys: {total_hobbys}")
    print(f"Total haves: {total_haves}")
    print(f"Total works: {total_works}")
    print(f"Total attitudes: {total_attitudes}")
    print(f"Total relationships: {total_relationships}")



    print(f"Total demographics: {total_demographics}")
    print(f"Total experiences: {total_experiences}")
    print(f"Total attitudes: {total_attitudes}")
    print(f"Total relationships: {total_relationships}")


    # Convert the data to pandas dataframes 
    identitydf = pd.DataFrame(identityList, columns = ["identity", "author_fullname"])
    agesdf = pd.DataFrame(agesList, columns = ["age", "author_fullname"])
    genderdf = pd.DataFrame(genderList, columns = ["gender", "author_fullname"])
    hobbysdf = pd.DataFrame(hobbysList, columns = ["hobby", "author_fullname"])
    havesdf = pd.DataFrame(havesList, columns = ["have", "author_fullname"])
    worksdf = pd.DataFrame(worksList, columns = ["work", "author_fullname"])
    attitudesdf = pd.DataFrame(attitudesList, columns = ["attitude", "author_fullname"])
    relationshipsdf = pd.DataFrame(relationshipsList, columns = ["relationship", "author_fullname"])    

    # Group the data by author
    identitydf = identitydf.groupby("author_fullname").agg(lambda x: x.tolist()).reset_index()
    # take the latest age only
    agesdf = agesdf.groupby("author_fullname").agg(lambda x: x.tolist()).reset_index()
    agesdf["age"] = agesdf["age"].apply(lambda x: x[-1] if x else None)
    # take the latest gender
    genderdf = genderdf.groupby("author_fullname").agg(lambda x: x.tolist()).reset_index()
    genderdf["gender"] = genderdf["gender"].apply(lambda x: x[-1] if x else None)
    hobbysdf = hobbysdf.groupby("author_fullname").agg(lambda x: x.tolist()).reset_index()
    havesdf = havesdf.groupby("author_fullname").agg(lambda x: x.tolist()).reset_index()
    worksdf = worksdf.groupby("author_fullname").agg(lambda x: x.tolist()).reset_index()
    attitudesdf = attitudesdf.groupby("author_fullname").agg(lambda x: x.tolist()).reset_index()
    relationshipsdf = relationshipsdf.groupby("author_fullname").agg(lambda x: x.tolist()).reset_index()

    # Merge the dataframes
    merged = pd.merge(identitydf, agesdf, on="author_fullname", how="outer")
    merged = pd.merge(merged, genderdf, on="author_fullname", how="outer")
    merged = pd.merge(merged, hobbysdf, on="author_fullname", how="outer")
    merged = pd.merge(merged, havesdf, on="author_fullname", how="outer")
    merged = pd.merge(merged, worksdf, on="author_fullname", how="outer")
    merged = pd.merge(merged, attitudesdf, on="author_fullname", how="outer")
    merged = pd.merge(merged, relationshipsdf, on="author_fullname", how="outer")

    merged['identity'] = merged['identity'].apply(lambda x: [] if isinstance(x, list) and not x else x if isinstance(x, list) else [] )
    merged['hobby'] = merged['hobby'].apply(lambda x: [] if isinstance(x, list) and not x else x if isinstance(x, list) else [] )
    merged['have'] = merged['have'].apply(lambda x: [] if isinstance(x, list) and not x else x if isinstance(x, list) else [] )
    merged['work'] = merged['work'].apply(lambda x: [] if isinstance(x, list) and not x else x if isinstance(x, list) else [] )
    merged['attitude'] = merged['attitude'].apply(lambda x: [] if isinstance(x, list) and not x else x if isinstance(x, list) else [] )
    merged['relationship'] = merged['relationship'].apply(lambda x: [] if isinstance(x, list) and not x else x if isinstance(x, list) else [] )

    # Save the data to a csv file
    merged.to_csv("data/regex/full_author_data.csv")

    labeled_dataset = pd.merge(labeled_data, merged, on="author_fullname", how="left")

  

    # Min-Max Normalization for age
    labeled_dataset["age"] = labeled_dataset["age"].astype(float)
    minAge = labeled_dataset["age"].min()
    maxAge = labeled_dataset["age"].max()
    labeled_dataset["age"] = labeled_dataset["age"].apply(lambda x: (float(x) - minAge) / (maxAge - minAge) if float(x) else None)

    # embedding for gender 
    gender_copy = labeled_dataset["gender"].copy()
    for i in range(len(gender_copy)):
            if isinstance(gender_copy[i], float):
                gender_copy[i] = ""
    all_values = [value for row in gender_copy for value in row]
    unique_genders = set(all_values)
    gender_vocab = {value: i for i, value in enumerate(unique_genders)}

    embedding_dim = 32
    embedding = nn.Embedding(len(gender_vocab), embedding_dim)

    embedded_rows = []
    for row in gender_copy:
        if not row:  # Handle empty lists
            embedded_rows.append(torch.zeros(embedding_dim))
            continue
        indices = torch.tensor([gender_vocab[item] for item in row])
        embedded = embedding(indices)
        mean_embedding = embedded.mean(dim=0)
        embedded_rows.append(mean_embedding)

    feature_matrix = torch.stack(embedded_rows)
    labeled_dataset["gender"] = feature_matrix.detach().numpy().tolist()


    # Create embeddings of identities, hobbies, haves, works, attitudes, and relationships
    def feature_embedding(data):
        copy = data.copy()
        # check for nan
        for i in range(len(copy)):
            if isinstance(copy[i], float):
                copy[i] = []

        # Create a dictionary of all unique values        
        all_values = [value for row in copy for value in row]
        unique_values = set(all_values)
        feature_vocab = {value: i for i, value in enumerate(unique_values)}

        embedding_dim = 32
        embedding = nn.Embedding(len(feature_vocab), embedding_dim)

        embedded_rows = []
        for row in copy:
            if not row:  # Handle empty lists
                embedded_rows.append(torch.zeros(embedding_dim))
                continue
            indices = torch.tensor([feature_vocab[item] for item in row])
            embedded = embedding(indices)
            mean_embedding = embedded.mean(dim=0)
            embedded_rows.append(mean_embedding)

        feature_matrix = torch.stack(embedded_rows)
        return feature_matrix.detach().numpy().tolist()

    # labeled_dataset["identity"] = feature_embedding(labeled_dataset["identity"])
    # labeled_dataset["hobby"] = feature_embedding(labeled_dataset["hobby"])
    # labeled_dataset["have"] = feature_embedding(labeled_dataset["have"])
    # labeled_dataset["work"] = feature_embedding(labeled_dataset["work"])
    # labeled_dataset["attitude"] = feature_embedding(labeled_dataset["attitude"])
    # labeled_dataset["relationship"] = feature_embedding(labeled_dataset["relationship"])

    # Save the labeled data to a csv file
    labeled_dataset.to_json("data/regex/labeled_data_short.json", orient="records", indent=4)

if __name__ == "__main__":
    main()
