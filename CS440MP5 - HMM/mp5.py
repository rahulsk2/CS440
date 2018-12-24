# mp5.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Renxuan Wang (renxuan2@illinois.edu) on 10/18/2018
import sys
import argparse

from reader import load_dataset, strip_tags
from viterbi import viterbi, baseline

"""
This file contains the main application that is run for this MP.
"""

'''
Evaluate output
input:  two lists of sentences with tags on the words
        one is predicted output, one is the correct tags
output: accuracy number (percentage of tags that match)
'''
def compute_accuracies(predicted_sentences, tag_sentences):
    correct = 0
    incorrect = 0
    count = 0
    for i in range(len(predicted_sentences)):
        for j in range(len(predicted_sentences[i])):
            count += 1
            if predicted_sentences[i][j][1] == tag_sentences[i][j][1]:
                correct += 1
            else:
                incorrect += 1
    return correct/(correct + incorrect)


def main(args):
    train_set = load_dataset(args.training_file, args.case_sensitive)
    test_set = load_dataset(args.test_file, args.case_sensitive)
    if args.baseline:
        print("You are running the baseline algorithm!")
        accuracy = compute_accuracies(test_set, baseline(train_set, strip_tags(test_set)))
    else:
        print("You are running the Viterbi algorithm!")
        accuracy = compute_accuracies(test_set, viterbi(train_set, strip_tags(test_set), args.alpha, args.beta))
    print("Accuracy:",accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CS440 MP5 HMM')
    parser.add_argument('--train', dest='training_file', type=str,
        help='the file of the training data')
    parser.add_argument('--test', dest='test_file', type=str,
        help='the file of the testing data')
    parser.add_argument('--case', dest='case_sensitive', default=False, action='store_true',
        help='Case sensitive (default false)')
    parser.add_argument('--baseline', dest='baseline', default=False, action='store_true',
        help='Use baseline algorithm')
    parser.add_argument('--viterbi', dest='viterbi', default=False, action='store_true',
        help='Use Viterbi algorithm')
    parser.add_argument('--alpha', dest='alpha', default=0.01, type=float,
                        help='Smoothing unseen words')
    parser.add_argument('--beta', dest='beta', default=15, type=float,
                        help='Smoothing unseen transitions')
    args = parser.parse_args()
    if args.training_file == None or args.test_file == None:
        sys.exit('You must specify training file and testing file!')
    if args.baseline ^ args.viterbi == False:
        sys.exit('You must specify using baseline or Viterbi!')

    main(args)
