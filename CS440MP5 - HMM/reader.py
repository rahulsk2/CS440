# reader.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Renxuan Wang (renxuan2@illinois.edu) on 10/18/2018


'''
Read training or test data
input:  filename
output: list of sentences
        each sentence is a list of (word,tag) pairs
        E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
'''
def load_dataset(data_file, case_sensitive = False):
    sentences = []
    with open(data_file, 'r', encoding='UTF-8') as f:
        for line in f:
            sentence = []
            raw = line.split()
            for pair in raw:
                splitted = pair.split('=')
                if case_sensitive:
                    sentence.append((splitted[0], splitted[1]))
                else:
                    sentence.append((splitted[0].lower(), splitted[1]))
            sentences.append(sentence)
    return sentences

'''
Strip tags
input:  list of sentences
        each sentence is a list of (word,tag) pairs
output: list of sentences
        each sentence is a list of words (no tags)
'''
def strip_tags(sentences):
    sentences_without_tags = []

    for sentence in sentences:
        sentence_without_tags = []
        for i in range(len(sentence)):
            pair = sentence[i]
            sentence_without_tags.append(pair[0])
        sentences_without_tags.append(sentence_without_tags)

    return sentences_without_tags
