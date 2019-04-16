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
    df = pd.read_csv(args.data_folder + 'cleaned_quotes.csv')
    quotes = df['Quotes'].values
    
    df_wordfreq = words_freq(quotes)
    df_wordfreq.to_csv(args.data_folder + 'word_count.csv')
    tokenizer = tokenize(df_wordfreq["word"].values[:15000], df["Quotes"].values, max_vocab=10000)
    token_quotes = quotes_to_token(quotes, tokenizer)
    assert len(token_quotes) == len(quotes)
    save_tokenizer(tokenizer,args.data_folder + 'tokenizer.pickle')
    np.save(args.data_folder+'train_quotes.npy',token_quotes[:400000])
    np.save(args.data_folder+'val_quotes.npy',token_quotes[400000:])
