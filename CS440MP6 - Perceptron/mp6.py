# mp6.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
import sys
import argparse
import configparser
import copy
import numpy as np

import reader
import perceptron as p

"""
This file contains the main application that is run for this MP.
"""

def compute_accuracies(predicted_labels,dev_set,dev_labels):
    yhats = predicted_labels
    accuracy = np.mean(yhats == dev_labels)
    tp = np.sum([yhats[i] == dev_labels[i] and yhats[i] == 1 for i in range(len(yhats))])
    precision = tp / np.sum([yhats[i]==1 for i in range(len(yhats))])
    recall = tp / (np.sum([yhats[i] != dev_labels[i] and yhats[i] == 0 for i in range(len(yhats))]) + tp)
    f1 = 2 * (precision * recall) / (precision + recall)

    return accuracy,f1,precision,recall

def main(args):
    train_set, train_labels, dev_set,dev_labels = reader.load_dataset(args.dataset_file)
    if not args.extra:
        predicted_labels = p.classify(train_set,train_labels, dev_set,args.lrate,args.max_iter)
    else:
        predicted_labels = p.classifyEC(train_set,train_labels,dev_set,args.lrate,args.max_iter)
    accuracy,f1,precision,recall = compute_accuracies(predicted_labels,dev_set,dev_labels)
    print("Accuracy:",accuracy)
    print("F1-Score:",f1)
    print("Precision:",precision)
    print("Recall:",recall)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CS440 MP6 Perceptron')

    parser.add_argument('--dataset', dest='dataset_file', type=str, default = 'mp6_data',
                        help='the directory of the training data')
    parser.add_argument('--extra',default=False,action="store_true",
                        help='Call extra credit function')
    parser.add_argument('--lrate',dest="lrate", type=float, default = 1.0,
                        help='Learning rate - default 1.0')
    parser.add_argument('--max_iter',dest="max_iter", type=int, default = 10,
                        help='Maximum iterations - default 10')

    args = parser.parse_args()
    main(args)
