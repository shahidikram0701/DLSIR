import json
import fasttext
import numpy as np
import pandas as pd
import os
#import tensorflow_hub as hub
import tensorflow.keras.layers as layers
from tensorflow.keras.datasets import imdb
from gensim.models import Word2Vec
from gensim.models import FastText
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split

def pprint(s):
  n = 35 - (len(s) // 2)
  print('-' * n, end = " ")
  print(s, end = " ")
  print('-' * n)

def json_to_dict(json_set):
    for k,v in json_set.items():
        if v == "True":
            json_set[k]= True
        elif v == "False":
            json_set[k]=False
        else:
            json_set[k]=v
    return json_set

def load_data(vocab_size,max_len):
    """
        Loads the keras imdb dataset
        Args:
            vocab_size = {int} the size of the vocabulary
            max_len = {int} the maximum length of input considered for paddingf
        Returns:
            X_train = tokenized train data
            X_test = tokenized test data
    """
    INDEX_FROM = 3

    (X_train,y_train),(X_test,y_test) = imdb.load_data(num_words = vocab_size,index_from = INDEX_FROM)

    return X_train,X_test,y_train,y_test


def prepare_data_for_word_vectors_imdb(X_train):
    """
        Prepares the input
        Args:
            X_train = tokenized train data
        Returns:
            sentences = {list} sentences containing words as tokens
            word_index = {dict} word and its indexes in whole of imdb corpus
    """
    INDEX_FROM = 3
    word_to_index = imdb.get_word_index()
    word_to_index = {k:(v+INDEX_FROM) for k,v in word_to_index.items()}

    word_to_index["<START>"] =1
    word_to_index["<UNK>"]=2

    index_to_word = {v:k for k,v in word_to_index.items()}

    sentences = []
    for i in range(len(X_train)):
        temp = [index_to_word[ids] for ids in X_train[i]]
        sentences.append(temp)
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    word_indexes = tokenizer.word_index
    """
    return sentences,word_to_index


def prepare_data_for_word_vectors(X):
    sentences_as_words=[]
    word_to_index={}
    count=1
    for sent in X:
        temp = sent.split()
        sentences_as_words.append(temp)
    for sent in sentences_as_words:
        for word in sent:
            if word_to_index.get(word,None) is None:
                word_to_index[word] = count
                count +=1
    index_to_word = {v:k for k,v in word_to_index.items()}
    sentences=[]
    for i in range(len(sentences_as_words)):
        temp = [word_to_index[w] for w in sentences_as_words[i]]
        sentences.append(temp)


    return sentences_as_words,sentences,word_to_index

def data_prep_ELMo(train_x,train_y,test_x,test_y,max_len):

    INDEX_FROM = 3
    word_to_index = imdb.get_word_index()
    word_to_index = {k:(v+INDEX_FROM) for k,v in word_to_index.items()}

    word_to_index["<START>"] =1
    word_to_index["<UNK>"]=2

    index_to_word = {v:k for k,v in word_to_index.items()}

    sentences=[]
    for i in range(len(train_x)):
        temp = [index_to_word[ids] for ids in train_x[i]]
        sentences.append(temp)

    test_sentences=[]
    for i in range(len(test_x)):
        temp = [index_to_word[ids] for ids in test_x[i]]
        test_sentences.append(temp)

    train_text = [' '.join(sentences[i][:max_len]) for i in range(len(sentences))]
    train_text = np.array(train_text, dtype=object)[:, np.newaxis]
    train_label = train_y.tolist()

    test_text = [' '.join(test_sentences[i][:500]) for i in range(len(test_sentences))]
    test_text = np.array(test_text , dtype=object)[:, np.newaxis]
    test_label = test_y.tolist()

    return train_text,train_label,test_text,test_label


def building_word_vector_model(option,sentences,embed_dim,workers,window,y_train):
    """
        Builds the word vector
        Args:
            type = {bool} 0 for Word2vec. 1 for gensim Fastext. 2 for Fasttext 2018.
            sentences = {list} list of tokenized words
            embed_dim = {int} embedding dimension of the word vectors
            workers = {int} no. of worker threads to train the model (faster training with multicore machines)
            window = {int} max distance between current and predicted word
            y_train = y_train
        Returns:
            model = Word2vec/Gensim fastText/ Fastext_2018 model trained on the training corpus
    """
    if option == 0:
        print("Training a word2vec model")
        model = Word2Vec(sentences=sentences, size = embed_dim, workers = workers, window = window)
        print("Training complete")

    elif option == 1:
        print("Training a Gensim FastText model")
        model = FastText(sentences=sentences, size = embed_dim, workers = workers, window = window)
        print("Training complete")

    elif option == 2:
        pprint("Training a Fasttext model from Facebook Research")
        y_train = ["__label__positive" if i==1 else "__label__negative" for i in y_train]
        # print("_" * 50)
        with open("imdb_train.txt","w") as text_file:
            for i in range(len(sentences)):
                print(sentences[i],y_train[i], file=text_file)
        # print("_" * 50)
        # print(help(fasttext.skipgram))
        model = fasttext.skipgram("imdb_train.txt","model_ft_2018_imdb",dim = embed_dim,min_count = 2)
        print("Training complete")

    return model

def padding_input(X_train,X_test,maxlen):
    """
        Pads the input upto considered max length
        Args:
            X_train = tokenized train data
            X_test = tokenized test data
        Returns:
            X_train_pad = padded tokenized train data
            X_test_pad = padded tokenized test data
    """

    X_train_pad = pad_sequences(X_train,maxlen=maxlen,padding="post")

    X_test_pad = pad_sequences(X_test,maxlen=maxlen,padding="post")

    return X_train_pad,X_test_pad

params_set = {
    "embed_dim":300,
    "split_ratio":0.33,
    "max_len":200,
    "vocab_size":10000,
    "trainable_param":"False",
    "option":2,
    "workers":3,
    "window":1
}

'''
    X = text data column
    y = label column(0,1 etc)

'''
if(not(os.path.exists("embedding_matrix.npy"))):
    if params_set["option"]in [0,1,2]:
        
        # for other data:
        
        X = ["this is a sentence","this is another sentence by me","yet another sentence for training","one more again"]
        y=np.array([0,1,1,0])

        sentences_as_words,sentences,word_ix = prepare_data_for_word_vectors(X)
        print("sentences loaded")
        model_wv = building_word_vector_model(params_set["option"],sentences,params_set["embed_dim"],
                                        params_set["workers"],params_set["window"],y)


        print("model built")
        x_train, x_test, y_train, y_test = train_test_split(sentences, y, test_size=params_set["split_ratio"], random_state=42)
        print("Data split done")
        
        x_train_pad,x_test_pad = padding_input(x_train,x_test,params_set["max_len"])
        #test if the embedding model loaded properly
        #print(model_wv['book'])
    

    else:
        x_train,x_test,y_train,y_test = load_data(params_set["vocab_size"],params_set["max_len"])

        train_text,train_label,test_text,test_label = data_prep_ELMo(x_train,y_train,x_test,y_test,params_set["max_len"])