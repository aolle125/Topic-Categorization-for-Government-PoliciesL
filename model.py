#Importing the Packages Required



from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer

from keras.wrappers.scikit_learn import KerasClassifier
import os
import re
import sys
import numpy as np
import pandas as pd

import keras

from keras.layers import Activation, Conv2D, Input, Embedding, Reshape, MaxPool2D, Concatenate, Flatten, Dropout, Dense, Conv1D
from keras.layers import MaxPool1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
import matplotlib.pyplot as plt


from keras.initializers import Constant
from keras.models import Model
from keras.layers import *
from keras import regularizers


#args.model should be: 0 for Static CNN, 1 for Non-Static CNN, 2 for Multi Channel CNN


def cnn1(MAX_WORDS,embedding_dim,MAX_SEQUENCE_LENGTH,lr,X_train,y_train,batch_size,
                  epochs,num_filters):
                  
    #A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.
    print("You have Chosen the Static CNN option")
    model = Sequential()        
    
    #Embeddings are low-dimensional, learned continuous vector representations of discrete variables
    
    model.add(layers.Embedding(input_dim=MAX_WORDS, output_dim=embedding_dim, input_length=MAX_SEQUENCE_LENGTH))
    
    #This layer creates a convolution kernel that is convolved with the layer input over a single spatial (or temporal) dimension to produce a tensor of outputs. 
    
    model.add(layers.Conv1D(num_filters, 3, activation='relu',padding='valid'))
    # perform max pooling
    
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(num_filters, activation='relu'))
    
    # do dropout and predict
    
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(18, activation='softmax'))
    
    #Compiling the model with the optimizers, loss and metrics
    model.compile(optimizer= Adam(lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_split = .10 , shuffle=True,
                   callbacks = [EarlyStopping(monitor='val_loss', patience=10)])
    model.save("model.h5")      #Saving the Model Architecture and Weights
    return history


def cnn2(MAX_WORDS,embedding_dim,MAX_SEQUENCE_LENGTH,lr,X_train,y_train,batch_size,
                  epochs,num_filters,num_words,embedding_matrix):
                  
    print("You have Chosen the Non-Static CNN option")
                  
    #A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.
    model = Sequential()
    
    #Embeddings are low-dimensional, learned continuous vector representations of discrete variables
    model.add(layers.Embedding(num_words,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True))
                                
    #This layer creates a convolution kernel that is convolved with the layer input over a single spatial (or temporal) dimension to produce a tensor of outputs.                            
    model.add(layers.Conv1D(num_filters, 3, activation='relu'))
    # perform max pooling
    
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(54, activation='relu'))
    
    # do dropout and predict
    
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(18, activation='softmax'))
    
    #Compiling the model with the optimizers, loss and metrics
    
    model.compile(optimizer= Adam(lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_split = .10 , shuffle=True,
                       callbacks = [EarlyStopping(monitor='val_loss', patience=10)])
    model.save("model.h5")      #Saving the Model Architecture and Weights
    return history

def cnn3(MAX_WORDS,embedding_dim,MAX_SEQUENCE_LENGTH,lr,X_train,y_train,batch_size,
                  epochs,num_filters,num_words,embedding_matrix):
    
    print("You have Chosen the Multi-Channel CNN option")
    
    inputs = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

    #Embeddings are low-dimensional, learned continuous vector representations of discrete variables
    embedding_layer = Embedding(num_words,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)(inputs)

    reshape = Reshape((MAX_SEQUENCE_LENGTH, embedding_dim, 1))(embedding_layer)

    #This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs.
    conv_0 = Conv2D(num_filters, kernel_size=(3, embedding_dim), activation='relu')(reshape)
    #conv_1 = Conv2D(num_filters, kernel_size=(4, embedding_dim), activation='relu')(reshape)
    conv_2 = Conv2D(num_filters, kernel_size=(5, embedding_dim), activation='relu')(reshape)

    # perform max pooling on each of the convoluations
    maxpool_0 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - 3 + 1, 1), strides=(1,1), padding='valid')(conv_0)
    #maxpool_1 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - 4 + 1, 1), strides=(1,1), padding='valid')(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - 5 + 1, 1), strides=(1,1), padding='valid')(conv_2)

    # concat and flatten
    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_2])
    flatten = Flatten()(concatenated_tensor)

    # do dropout and predict
    dropout = Dropout(0.5)(flatten)
    output = Dense(units=18, activation='softmax')(dropout)

    model = Model(inputs=inputs, outputs=output)
    
    #Compiling the model with the optimizers, loss and metrics
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_split = .10 , shuffle=True,
                       callbacks = [EarlyStopping(monitor='val_loss', patience=10)])
                      
    model.save("model.h5")      #Saving the Model Architecture and Weights

    return history