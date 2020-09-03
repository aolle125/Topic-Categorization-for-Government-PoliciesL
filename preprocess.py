#Importing the Packages Required


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os
import re
import sys
import numpy as np
import pandas as pd

import keras
from nltk.stem import WordNetLemmatizer 
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical


import matplotlib.pyplot as plt

from collections import Counter
import nltk

from args import args

from nltk.corpus import stopwords

#Adding manual stop words which are not important in classification 

stop = stopwords.words('english')
stop.extend(['introduced','virginia', 'bill', 'session', 'person' , 'house', 'no','author', 'license', 'local','public','article'
        'city','county','agency', 'amends','chapter','act','amend','reenact','code','approved','pursuant','designated','chapter',
        'rule','exercise','criterion','town','subject','prelude','shall','rule','permitted','restricted','regulation',
        'assembly','state'])


#Function for converting labels to Categorical, for output in the Classifier

def encode(Y):
    print("Converting labels to Categorical")
    le = LabelEncoder()
    Y = le.fit_transform(Y)
    Y1 = le.fit_transform(Y)
    Y = to_categorical(Y)
    Y1 = np.argmax(Y, axis = 1)
    return Y,Y1



#Cleaning of the summary for valid inputs

def process(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)      
    return string.strip().lower()


#Finding the maximum length summary

def max_length(df,X):
    df['l'] = df['summary'].apply(lambda x: len(str(x).split(' ')))
    return df.l.max()

#Tokenization and Padding of Input Summaries

def token(X,MAX_SEQUENCE_LENGTH):
    print("Tokenization and Padding of Input Summaries")
    tokenizer  = Tokenizer(num_words = args.MAX_WORDS)
    tokenizer.fit_on_texts(X)
    X = tokenizer.texts_to_sequences(X)
    X = pad_sequences(X, MAX_SEQUENCE_LENGTH)
    return X,tokenizer

#Converting the input into GloVe embeddings

def glove(tokenizer,embedding_dim,MAX_WORDS):

    print("Finding GloVe Embeddings")
    embeddings_index = {}
    f = open('glove.6B.100d.txt',encoding='utf8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Total %s word vectors in Glove 6B 100d.' % len(embeddings_index))
    word_index = tokenizer.word_index
    num_words = len(word_index) + 1
    # first create a matrix of zeros, this is our embedding matrix
    embedding_matrix = np.zeros((num_words, embedding_dim))

    # for each word in out tokenizer lets try to find that work in our w2v model
    for word, i in word_index.items():
        if i > MAX_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # we found the word - add that words vector to the matrix
            embedding_matrix[i] = embedding_vector
        else:
            # doesn't exist, assign a random vector
            embedding_matrix[i] = np.random.randn(embedding_dim)
    return embedding_matrix,num_words


#Stop Word removal

def stopword(df):
    df.summary = df.summary.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    return df


#Lemmatization of input

def lemma(df):
    lemmatizer = WordNetLemmatizer() 
    df.summary = df.summary.apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()] ))
    return df


#Undersampling Classes

def undersample(df):
    df['topic'] = Y1
    num = [i for i in range(max(Y) + 1)]
    df = df.loc[df['topic'].isin(num)].groupby('majortopic').head(53)       #Choose a number of your choice
    return df

#Oversampling Classes

def duplicate(df):
    data = df.loc[df['topic']=="Technology"]
    df = df.append(data)
    return df

