# viterbi.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Renxuan Wang (renxuan2@illinois.edu) on 10/18/2018
import math


list_of_tags = ['NOUN', 'CONJ', 'VERB', 'ADJ', 'ADP', 'X', 'PRON', '.', 'ADV', 'DET', 'PRT', 'NUM']

def convert_counts_to_probabilities(counts_map, smoothing_factor):
    result = {}
    for each_outer_entry_key, each_outer_entry_values in counts_map.items():
        total_values = len(each_outer_entry_values) + 1 #1 for UNK
        normalizer = sum(each_outer_entry_values.values()) + (smoothing_factor * total_values)
        inner_map = {}
        for each_inner_entry_key, each_inner_entry_value in each_outer_entry_values.items():
            normalized_value = (each_inner_entry_value + smoothing_factor) / normalizer
            inner_map[each_inner_entry_key] = math.log10(normalized_value)
        inner_map["UNK"] = math.log10(smoothing_factor / normalizer)
        result[each_outer_entry_key] = inner_map
    return result


def build_datapoints(train):
    word_to_class_counts = {}
    length_to_tag = {}
    transition_counts = {}
    class_to_word_counts = {}
    transition_counts["FINAL"] = {}
    for each_line in train:
        previous_tag = "START"
        for each_word, each_tag in each_line:
            #1. Emission Probabilities
            if each_word in word_to_class_counts:
                each_word_map = word_to_class_counts.get(each_word)
                each_word_map[each_tag] = each_word_map.get(each_tag, 0) + 1
                word_to_class_counts[each_word] = each_word_map
            else:
                word_to_class_counts[each_word] = {each_tag:1}
            #3. Most commonly seen tag
            length = len(each_word)
            if length in length_to_tag:
                len_map = length_to_tag.get(length)
                len_map[each_tag] = len_map.get(each_tag, 0) + 1
                length_to_tag[length] = len_map
            else:
                length_to_tag[length] = {each_tag: 1}

            if each_tag in class_to_word_counts:
                each_tag_map = class_to_word_counts.get(each_tag)
                each_tag_map[each_word] = each_tag_map.get(each_word, 0) + 1
                class_to_word_counts[each_tag] = each_tag_map
            else:
                class_to_word_counts[each_tag] = {each_word: 1}

            #2. Transition Probabilities
            if each_tag in transition_counts:
                each_tag_map = transition_counts.get(each_tag)
                each_tag_map[previous_tag] = each_tag_map.get(previous_tag, 0) + 1
                transition_counts[each_tag] = each_tag_map
                previous_tag = each_tag
            else:
                transition_counts[each_tag] = {previous_tag:1}
                previous_tag = each_tag
        tag_map_final = transition_counts["FINAL"]
        tag_map_final[previous_tag] = tag_map_final.get(previous_tag, 0) + 1
        transition_counts["FINAL"] = tag_map_final

    return word_to_class_counts, transition_counts, length_to_tag, class_to_word_counts

def build_emission_probabilities(train):
    emission_probabilities = {}
    for each_line in train:
        for each_word, each_tag in each_line:
            if each_word in emission_probabilities:
                each_word_map = emission_probabilities.get(each_word)
                each_word_map[each_tag] = each_word_map.get(each_tag, 0) + 1
                emission_probabilities[each_word] = each_word_map
            else:
                emission_probabilities[each_word] = {each_tag:1}
    return emission_probabilities


def build_baseline_prediction(test, emission_probabilities, length_to_tags):
    predicts = []
    for each_line in test:
        line = []
        for each_word in each_line:
            if each_word in emission_probabilities:
                tag = max(emission_probabilities[each_word], key=emission_probabilities[each_word].get)
                line.append((each_word, tag))
            else:
                length = len(each_word)
                if length in length_to_tags:
                    tag = max(length_to_tags[length], key=length_to_tags[length].get)
                    line.append((each_word, tag))
                else:
                    line.append((each_word, "UNSEEN"))
        predicts.append(line)
    return predicts


class viterbi_trellis_cell:
    def __init__(self, value, previous_tag):
        self.value = value
        self.previous_tag = previous_tag

    def __lt__(self, other):
        return self.value < other.value

    def __str__(self):
        return "Value:" + str(self.value) + " Previous:" + self.previous_tag

    def __eq__(self, other):
        return self.value == other.value


"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

'''
TODO: implement the baseline algorithm.
input:  training data (list of sentences, with tags on the words)
        test data (list of sentences, no tags on the words)
output: list of sentences, each sentence is a list of (word,tag) pairs. 
        E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
'''
def baseline(train, test):
    word_to_class_counts, transition_counts, length_to_tags, class_to_word_counts = build_datapoints(train)
    predicts = build_baseline_prediction(test, word_to_class_counts, length_to_tags)
    return predicts


def run_viterbi_algorithm(each_line, class_to_word_probabilities, transition_probabilities, length_to_tags):
    no_of_words = len(each_line)
    no_of_tags = len(list_of_tags)
    viterbi_trellis = [[viterbi_trellis_cell(0, 'NONE') for y in range(no_of_words)] for x in range(no_of_tags)]
    line = []
    for index, each_tag in enumerate(list_of_tags):
        if each_line[0] in class_to_word_probabilities[each_tag]:
            p_w_given_tag = class_to_word_probabilities[each_tag].get(each_line[0])
        else:
            p_w_given_tag = class_to_word_probabilities[each_tag].get("UNK")
        if 'START' in transition_probabilities[each_tag]:
            p_tag_given_start = transition_probabilities[each_tag].get('START')
        else:
            p_w_given_tag = transition_probabilities[each_tag].get("UNK")
        viterbi_trellis[index][0] = viterbi_trellis_cell(p_w_given_tag + p_tag_given_start, 'START')

    for index, each_word in enumerate(each_line):
        if index == 0:
            continue
        for i, each_current_tag in enumerate(list_of_tags):
            max_possible = -99999999999999999
            back_pointer = 'NONE'
            for j, each_previous_tag in enumerate(list_of_tags):
                previous_best_of_tag = viterbi_trellis[j][index-1].value
                if each_previous_tag in transition_probabilities[each_current_tag]:
                    transition_from_prev_tag = transition_probabilities[each_current_tag].get(each_previous_tag)
                else:
                    transition_from_prev_tag = transition_probabilities[each_current_tag].get("UNK")
                combined_value = previous_best_of_tag + transition_from_prev_tag
                if combined_value > max_possible:
                    back_pointer = each_previous_tag
                    max_possible = combined_value
            if each_word in class_to_word_probabilities[each_current_tag]:
                p_w_given_tag = class_to_word_probabilities[each_current_tag].get(each_word)
            else:
                p_w_given_tag = class_to_word_probabilities[each_current_tag].get("UNK")
            current_best_value = p_w_given_tag + combined_value
            viterbi_trellis[i][index] = viterbi_trellis_cell(current_best_value, back_pointer)

    last_tag = 'NONE'
    max_possible = -9999999999999999
    for index, each_tag in enumerate(list_of_tags):
        current_best_value = viterbi_trellis[index][-1].value
        # transition_to_final = math.log10((transition_counts['FINAL'].get(each_tag, 0) + beta) / (sum(transition_counts['FINAL'].values()) + beta * no_of_tags))
        if each_tag in transition_probabilities['FINAL']:
            transition_to_final = transition_probabilities['FINAL'].get(each_tag)
        else:
            transition_to_final = transition_probabilities['FINAL'].get("UNK")
        total_possible = current_best_value + transition_to_final
        if total_possible > max_possible:
            max_possible = total_possible
            last_tag = each_tag
    tags_list = []
    index = no_of_words - 1
    curr_tag = last_tag
    while curr_tag != 'START':
        tags_list.append(curr_tag)
        curr_tag = viterbi_trellis[list_of_tags.index(curr_tag)][index].previous_tag
        index -= 1
    tags_list = list(reversed(tags_list))
    for index, each_word in enumerate(each_line):
        line.append((each_word, tags_list[index]))
    return line




def build_viterbi_predictions(test, word_to_class_counts, transition_counts, length_to_tags):
    result = []
    for each_line in test:
        if each_line:
            tagged_line = run_viterbi_algorithm(each_line, word_to_class_counts, transition_counts, length_to_tags)
            result.append(tagged_line)
        else:
            tagged_line = []
            result.append(tagged_line)
    return result



'''
TODO: implement the Viterbi algorithm.
input:  training data (list of sentences, with tags on the words)
        test data (list of sentences, no tags on the words)
output: list of sentences with tags on the words
        E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
'''
def viterbi(train, test, alpha=0.01, beta=15):
    word_to_class_counts, transition_counts, length_to_tags, class_to_word_counts = build_datapoints(train)
    class_to_word_probabilities = convert_counts_to_probabilities(class_to_word_counts, alpha)
    transition_probabilities = convert_counts_to_probabilities(transition_counts, beta)
    predicts = build_viterbi_predictions(test, class_to_word_probabilities, transition_probabilities,length_to_tags)
    return predicts
