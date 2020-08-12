The task of the text classifier was to learn a model using the given training data that consisted of the summaries of the VA bills and policies and classify them into one of the eighteen categories that are given. There are two models that are discussed in the report. The first one is the original classifier earlier developed using Long-Short Term Memory Networks [1] and the second one is the newer version I have developed using Convolutional Neural Networks [2]. 
There are some common preprocessing steps taken in both the classifiers developed. The Pandas dataframe is used to store the two columns, one is the summaries of the bills and the other is the topic in which that summary belongs. The summary text is tokenized which means that all the words are broken down into a series of tokens. After this, we encode the topic i.e. the label into one-hot encoders as required in Keras for the output layer.

Firstly, the classifier used an embedding layer provided by Keras which learns a vector embedding for all the words in the dataset. 
Secondly, a pre-trained word embedding was learnt using GloVe [4] which stands for Global Vectors for Word Representation. GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space. 


Convolutional Neural Networks are state-of-the-art models when it comes to text classification. Convolutional neural networks are effective at document classification, namely because they can pick out salient features (e.g. tokens or sequences of tokens) in a way that is invariant to their position within the input sequences.
There are three types of CNNs used for text classification:
CNN-static : The embedding layer has trainable parameter set to False
CNN-non-static: The embedding layer has trainable parameter set to True
CNN-multichannel: The CNNs have different channels, where different types of filters are used on the embedding layer and their results are concatenated to produce the classification output.


K-Fold Cross Validation was used to split the train-test data. The average accuracy of all the folds was considered as the final testing accuracy. The maximum testing accuracy achieved by this classifier was 89.64%.
















