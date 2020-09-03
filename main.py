#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasClassifier
import os
import re
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import keras
from nltk.stem import WordNetLemmatizer 
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Activation, Conv2D, Input, Embedding, Reshape, MaxPool2D, Concatenate, Flatten, Dropout, Dense, Conv1D
from keras.layers import MaxPool1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam

from args import args
import matplotlib.pyplot as plt


from collections import Counter
import nltk

from keras.initializers import Constant
from keras.models import Model
from keras.layers import *
from keras import regularizers

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from preprocess import encode,max_length,process,token,glove,stopword,lemma,undersample,duplicate


# In[3]:


df = pd.read_csv("pa_sample.csv")


# In[5]:


Y = df.majortopic.values


# In[6]:


Y, Y1 = encode(Y)


# In[7]:


df.summary = df.summary.apply(lambda x: ' '.join([process(word) for word in x.split()] ))


# In[8]:


if args.s == 1:
    df = stopword(df)
if args.l == 1:
    df = lemma(df)
if args.u == 1:
    df = undersample(df)
if args.o == 1:
    df = oversample(df)


# In[9]:


X = df.summary.values


# In[10]:


MAX_SEQUENCE_LENGTH = max_length(df,X)


# In[11]:


X,tokenizer = token(X,MAX_SEQUENCE_LENGTH)


# In[12]:


X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.10)


# In[13]:


from model import cnn1,cnn2,cnn3


# In[14]:


embedding_matrix,num_words = glove(tokenizer,args.embedding_dim,args.MAX_WORDS)


# In[16]:


if args.model == 0:
    history = cnn1(args.MAX_WORDS,args.embedding_dim,MAX_SEQUENCE_LENGTH,args.lr,X_train,y_train,args.batch_size,
                  args.epochs,args.num_filters)
if args.model == 1:
    
    history = cnn2(args.MAX_WORDS,args.embedding_dim,MAX_SEQUENCE_LENGTH,args.lr,X_train,y_train,args.batch_size,
                  args.epochs,args.num_filters,num_words,embedding_matrix)
if args.model == 2:
    history = cnn3(args.MAX_WORDS,args.embedding_dim,MAX_SEQUENCE_LENGTH,args.lr,X_train,y_train,args.batch_size,
                  args.epochs,args.num_filters,num_words,embedding_matrix)


# In[17]:


from pred import pred,report,plot


# In[18]:


y_test1,pred = pred(X_test,y_test)


# In[19]:


report(y_test1,pred)


# In[20]:


history_dict = history.history
plot(history_dict)


# In[ ]:




