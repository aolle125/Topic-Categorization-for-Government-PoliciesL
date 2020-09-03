#Importing the Packages Required

import os
import re
import sys
import numpy as np
import pandas as pd

import keras

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report


from keras.models import load_model

#Predicting the output

def pred(X_test,y_test):
    print("Predicting the output")
    model = load_model('model.h5')
    pred = model.predict_classes(X_test, verbose=1)
    y_test1 = np.argmax(y_test, axis = 1)
    return y_test1,pred


#Displaying the Classification Report with Precision, Recall and F1-Score

def report(y_test1,pred):
    print(classification_report(y_test1, pred,labels = [i for i in range(18)]))



#Plotting the train vs val loss and train vs val accuracies of the model

def plot(history_dict):
    history_dict.keys()
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    plt.show()




