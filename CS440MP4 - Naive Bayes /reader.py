# reader.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
"""
This file is responsible for providing functions for reading the files
"""
from os import listdir
import numpy as np
from nltk.stem.porter import PorterStemmer

porter_stemmer = PorterStemmer()
bad_words = {'aed','oed','eed'} # these words fail in nltk stemmer algorithm
def loadDir(name,stemming):
    # Loads the files in the folder and returns a list of lists of words from
    # the text in each file
    X0 = []
    count = 0
    for f in listdir(name):
        fullname = name+f
        text = []
        with open(fullname, 'rb') as f:
            for line in f:
                text += line.decode(errors='ignore').split(' ')
        if stemming:
            for i in range(len(text)):
                if text[i] in bad_words:
                    continue
                text[i] = porter_stemmer.stem(text[i])
        X0.append(text)
        count = count + 1
    return X0

def load_dataset(train_dir,dev_dir,stemming):
    X0 = loadDir(train_dir + '/ham/',stemming)
    X1 = loadDir(train_dir + '/spam/',stemming)
    X = X0 + X1
    Y = len(X0) * [0] + len(X1) * [1]
    Y = np.array(Y)

    X_test0 = loadDir(dev_dir + '/ham/',stemming)
    X_test1 = loadDir(dev_dir + '/spam/',stemming)
    X_test = X_test0 + X_test1
    Y_test = len(X_test0) * [0] + len(X_test1) * [1]
    Y_test = np.array(Y_test)

    return X,Y,X_test,Y_test
