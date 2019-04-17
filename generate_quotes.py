import argparse
import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from model.simplemodel import simplemodel
import pickle


def generate_text(start_word, tokenizer, model, max_len = 816):
    output = tokenizer.texts_to_sequences([start_word])[0]
    endtoken = tokenizer.word_index['endtoken']

    for i in range(100):
        token_list = pad_sequences([output], maxlen=max_len, padding='pre')
        predicted = np.argmax(model.predict(token_list)[0][2:])+2
        if predicted == endtoken:
            return tokenizer.sequences_to_texts([output])
        output.append(predicted)
    return tokenizer.sequences_to_texts([output])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', default = '/home/angps/Documents/Quotes_generation/data/tokenizer.pickle')
    parser.add_argument('--model', default = '/home/angps/Documents/Quotes_generation/model_logs/model.hdf5')
    parser.add_argument('--startword', required=True, type=str)
    args = parser.parse_args()
    with open(args.token,'rb') as f:
        tokenizer = pickle.load(f)
    model = simplemodel()
    model.load_weights(args.model)
    print(generate_text(args.startword,tokenizer,model))