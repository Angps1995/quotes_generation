import numpy as np
import pandas as pd
import string
import os
from collections import Counter 
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from itertools import repeat
from keras.preprocessing.text import Tokenizer
import argparse
from utils import load_h5_data, save_as_h5, save_tokenizer, load_tokenizer
txt = pd.read_csv('/home/angps/Documents/Quotes_generation/data/cleaned_quotes.csv')


def add_start_end(texts):
    texts = 'starttoken ' + texts + ' endtoken'
    return (texts,)

def get_df_id_captions(txt_file):
    """ add start and end tokens. """

    pool = Pool(cpu_count() - 2)
    quotes = txt_file["Quotes"].values
    results = []
    for result in tqdm(pool.starmap(
        add_start_end, zip(
            quotes))):
        results.append(result)
    txt = []
    for i,r in enumerate(results):
        txt.append(r[0])
    
    df = pd.DataFrame()
    df["Quotes"] = txt
    return df

def words_freq(quotes):
    """ get a dataframe of words used"""

    vocab = []
    for txt in tqdm(quotes):
        vocab.extend(txt.split())
    print('Vocabulary Size: %d' % len(set(vocab)) + ' unique words')
    ct = Counter(vocab)
    dfword = pd.DataFrame({"word":list(ct.keys()),"count":list(ct.values())})
    dfword = dfword.sort_values("count",ascending=False)
    dfword = dfword.reset_index()[["word","count"]]
    return dfword

def tokenize(words, quotes, max_vocab=10000):
    """Tokenize the quotes"""

    tokenizer = Tokenizer(num_words = max_vocab,oov_token='<unk>')  #replace the least common words with '<unk>'
    tokenizer.fit_on_texts(words)
    vocab_size = len(tokenizer.word_index) + 1
    print("vocabulary size : {}".format(vocab_size))
    return tokenizer

def quotes_to_token(quotes, tokenizer):
    token_caps = tokenizer.texts_to_sequences(quotes)
    return token_caps

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', default = '/home/angps/Documents/Quotes_generation/data/')
    args = parser.parse_args()
    df = get_df_id_captions(txt)
    quotes = df['Quotes'].values
    df_wordfreq = words_freq(quotes)
    df_wordfreq.to_csv('cleaned2.csv')
    tokenizer = tokenize(df_wordfreq["word"].values[:20000], df["Quotes"].values, max_vocab=10000)
    token_quotes = np.array(quotes_to_token(df["Quotes"].values, tokenizer))
    assert len(token_quotes) == len(df['Quotes'].values)
    save_tokenizer(tokenizer,args.data_folder + 'tokenizer.pickle')
    np.save(args.data_folder+'tokenized_quotes.npy',token_quotes)
    print(token_quotes.shape)