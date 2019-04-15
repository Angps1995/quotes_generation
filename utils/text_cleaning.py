import numpy as np
import pandas as pd
import string
import os
from collections import Counter 
import tqdm
from multiprocessing import Pool,cpu_count

df = pd.read_csv('/home/angps/Documents/Quotes_generation/data/quotes.csv')
def add_space_after_punc(text):
    punc = [',','.',';',':','?','!']
    new = ''
    for c in range(len(text)-1):
        if text[c] in punc and text[c+1]!=' ':
            new+=text[c]+' '
        elif text[c] in punc and text[c+1]==' ':
            new+=text[c]
            c+=1
        else:
            new+=text[c]
    return new + text[-1]

def remove_single_character(text):
    len_more_than_one= ""
    words = text.split()
    for w in range(len(words)):
        if len(words[w]) > 1 and words[w]!='I'.lower():
            if w==0:
                len_more_than_one += words[w]
            else:
                len_more_than_one+= " " + words[w]
    return(len_more_than_one)

def text_clean(df):
    for i, quotes in enumerate(df["Quotes"].values):
        df["Quotes"].iloc[i] = add_space_after_punc(quotes.lower())
        df["Quotes"].iloc[i] = remove_single_character(quotes)
        df["Quotes"].iloc[i] = quotes.lower()
        if quotes[-1]==".":
            df["Quotes"].iloc[i] = df["Quotes"].iloc[i][:-1]
    return df
def parallelize_dataframe(df, func):
    num_cores = cpu_count()-2  
    num_partitions = num_cores #number of partitions to split dataframe
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


df = parallelize_dataframe(df,text_clean)
df.to_csv('cleaned_quotes.csv')






