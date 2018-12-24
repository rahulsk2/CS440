# perceptron.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
import numpy as np
import random
import time
"""
This is the main entry point for MP6. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


class Perceptron:
    def __init__(self, len_of_features, max_iter, learning_rate, skip=0):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.weights = np.zeros(len_of_features + 1) #weights[0] will hold the bias
        self.skip = skip

    def train(self, train_set, train_labels):
        """
        This method actually trains the perceptron and sets the self.weights
        We use the sign function for perceptron training.
        new_weight += learning_rate * (label - prediction) * current_weight
        :param train_set:
        :param train_labels:
        :return:
        """
        for epoch in range(1, self.max_iter + 1):
            if self.skip != 0:
                if self.skip == 1 and ((epoch+1) % 2) == 0:
                    continue
                if (epoch % self.skip) == 0:
                    continue
            for features, label in zip(train_set, train_labels):
                prediction = self.predict(features)
                self.weights[1:] += self.learning_rate * (label - prediction) * features #regular weights
                self.weights[0] += self.learning_rate * (label - prediction) * 1 #bias

    def predict(self, features):
        if (np.dot(features, self.weights[1:]) + self.weights[0]) > 0:
            return 1
        return 0

    def get_weights(self):
        return self.weights


def classify(train_set, train_labels, dev_set, learning_rate,max_iter):
    """
    train_set - A Numpy array of 32x32x3 images of shape [7500, 3072].
                This can be thought of as a list of 7500 vectors that are each
                3072 dimensional.  We have 3072 dimensions because there are
                each image is 32x32 and we have 3 color channels.
                So 32*32*3 = 3072
    train_labels - List of labels corresponding with images in train_set
    example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
             and X1 is a picture of a dog and X2 is a picture of an airplane.
             Then train_labels := [1,0] because X1 contains a picture of an animal
             and X2 contains no animals in the picture.

    dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
              It is the same format as train_set
    """
    # TODO: Write your code here
    # return predicted labels of development set

    start_time = time.time()
    perceptron = Perceptron(len(train_set[0]), max_iter, learning_rate, 0)
    perceptron.train(train_set, train_labels)
    print("Training Done in %s seconds." % (time.time() - start_time))
    dev_labels = []
    for each_image in dev_set:
        result = perceptron.predict(each_image)
        dev_labels.append(result)
    print("End to End Done in %s seconds." % (time.time() - start_time))
    return dev_labels


##########################
# Decision Tree
##########################

def classifyEC(train_set, train_labels, dev_set,learning_rate,max_iter):
    # Write your code here if you would like to attempt the extra credit
    start_time = time.time()
    train_set_temp1 = train_set[:50]
    train_set_temp2 = train_set[len(train_set)-50:]
    random_list = random.sample(range(51, len(train_set) - 51), 50)
    train_set_temp_3 = create_list(train_set, random_list)
    train_set = train_set_temp1 + train_set_temp2 + train_set_temp_3
    train_labels_temp1 = train_labels[:50]
    train_labels_temp2 = train_labels[len(train_labels)-50:]
    train_labels_temp3 = create_list(train_labels, random_list)
    train_labels = train_labels_temp1 + train_labels_temp2 + train_labels_temp3
    value_map = {True: 1.0, False: 0.0}
    train_a = np.array(train_set)
    train_b = np.array([[value_map[i] for i in train_labels]])
    train_c = np.concatenate((train_a, train_b.T), axis=1)
    dev_a = np.array(dev_set)
    dev_b = np.array([[None for i in range(len(dev_set))]])
    dev_c = np.concatenate((dev_a, dev_b.T), axis=1)
    predictions = decision_tree(train_c, dev_c, 5, 10)
    rev_value_map = {1.0: True, 0.0: False}
    dev_labels = [rev_value_map[i] for i in predictions]
    print("End to End Done in %s seconds." % (time.time() - start_time))
    return dev_labels

#THIS FOLLOWS THE CART ALGORITHM TO BUILD THE DECISION TREE
def decision_tree(train, dev, max_depth, min_size):
    tree = build_tree(train, max_depth, min_size)
    predictions = list()
    for row in dev:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return predictions

def build_tree(train, max_depth, min_size):
    root = get_split_points(train)
    split_data(root, max_depth, min_size, 1)
    return root

def get_split_points(train):
    class_values = list(set(row[-1] for row in train))
    # print("Class Values:", class_values)
    # print("Dataset:", dataset)
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(train[0]) - 1):
        for row in train:
            groups = test_split(index, row[index], train)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


def split_data(node, max_depth, min_size, depth):
    left, right = node['groups']
    del (node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split_points(left)
        split_data(node['left'], max_depth, min_size, depth + 1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split_points(right)
        split_data(node['right'], max_depth, min_size, depth + 1)


def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


def gini_index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini

def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


##########################
# Alt Idea
##########################


def create_list(first_list, second_list):
    result = []
    for i in second_list:
        result.append(first_list[i])
    return result


def get_single_value(input_list):
    return max(set(input_list), key=input_list.count)

def classifyECalt(train_set, train_labels, dev_set, learning_rate, max_iter):
    start_time = time.time()
    perceptron_inputs = []
    for i in range(0, 5):
        perceptron = Perceptron(len(train_set[0]), max_iter+i, learning_rate+i, skip=i)
        perceptron.train(train_set, train_labels)
        perceptron_inputs.append(perceptron)
    print("Training Done in %s seconds." % (time.time() - start_time))
    dev_labels = []
    for each_image in dev_set:
        first = []
        for i in range(5):
            perceptron = perceptron_inputs[i]
            answer = perceptron.predict(each_image)
            first.append(answer)
        second = list()
        second.append(get_single_value(first[0:3]))
        second.append(get_single_value(first[2:5]))
        second.append(get_single_value(first[::2]))
        second.append(get_single_value(first[1::2]))
        random_list = random.sample(range(0, 4), 3)
        second.append(get_single_value(create_list(first, random_list)))
        final_value = get_single_value(second)
        dev_labels.append(final_value)
    print("End to End Done in %s seconds." % (time.time() - start_time))
    return dev_labels
