
import gzip
import pandas as pd

def gzip_to_csv(file):
    # Load the gzip file
    with gzip.open(file, 'rb') as f:
        data = pd.read_pickle(f)
    
    # Save the data to csv
    data.to_csv(file[:-5] + '.csv', index=False)    


gzip_to_csv('data/social_comments_filtered.gzip')