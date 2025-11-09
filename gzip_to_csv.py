"""Convert gzip-compressed pickle files to CSV format.

This module provides utilities to convert compressed data files (gzip format
containing pandas pickle data) to CSV format for easier viewing and analysis.
"""

import gzip
import pandas as pd


def gzip_to_csv(file):
    """Convert a gzip-compressed pickle file to CSV format.
    
    Parameters
    ----------
    file : str
        Path to the gzip file containing pickled pandas DataFrame.
        The output CSV will have the same name with '.csv' extension.
    """
    # Load the gzip file
    with gzip.open(file, 'rb') as f:
        data = pd.read_pickle(f)
    
    # Save the data to csv
    data.to_csv(file[:-5] + '.csv', index=False)    


gzip_to_csv('data/social_comments_filtered.gzip')