#Importing the Packages

import os
import argparse
import datetime



parser = argparse.ArgumentParser(description='CNN Text Classification')

# PreProcessing
parser.add_argument('-s', type=int, default=0, help='Stop Word Removal Yes/No')
parser.add_argument('-l', type=int, default=0, help='Lemmatize Yes/No')
parser.add_argument('-u', type=int, default=0, help='Undersampling Yes/No')
parser.add_argument('-o', type=int, default=0, help='Oversampling Yes/No')

# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=100, help='number of epochs for train [default: 100]')
parser.add_argument('-batch-size', type=int, default=32, help='batch size for training [default: 32]')
parser.add_argument('-num_filters', type=int, default=64, help=' number of filters [default: 64]')
parser.add_argument('-embedding_dim', type=int, default=100, help='number of embedding dimension [default: 100]')
parser.add_argument('-MAX_WORDS', type=int, default=10000, help=' Maximum words for tokenization[default: 10000]')

# Model Type
parser.add_argument('-model', type=int, default=0, help='Choose the model, static[0],non-static[1],multi-channel[2]')

args, unknown = parser.parse_known_args()






