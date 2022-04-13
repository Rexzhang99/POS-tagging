# mp4.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created Fall 2018: Margaret Fleck, Renxuan Wang, Tiantian Fang, Edward Huang (adapted from a U. Penn assignment)
# Modified Spring 2020: Jialu Li, Guannan Guo, and Kiran Ramnath
# Modified Fall 2020: Amnon Attali, Jatin Arora
# Modified Spring 2021 by Kiran Ramnath
"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

from collections import Counter


def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    def max_key(dict):
        return max((dict).keys(), key=lambda key: dict[key])

    word_tag = {}
    tag_ct = Counter()
    for sentence in train:
        for word, tag in sentence:
            if word not in word_tag:
                word_tag[word] = Counter()
            word_tag[word][tag] += 1
            tag_ct[tag] += 1

    t_max = max_key(tag_ct)
    output = []
    for sentence in test:
        sentence_pred = []
        for word in sentence:
            if word in word_tag:
                sentence_pred.append((word, max_key(word_tag[word])))
            else:
                sentence_pred.append((word, t_max))
        output.append(sentence_pred)
    return output
