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
# Modified Spring 2021 by Kiran Ramnath (kiranr2@illinois.edu)

"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""
from numpy import argmax, log
import numpy as np
import cProfile
from collections import Counter, defaultdict
import time
# import pandas as pd


class HMM:
    def __init__(self, train, test) -> None:
        self.train = train
        self.test = test
        self.laplace = 1e-5
        self.tags, self.tag_pairs, self.word_tag_pairs = self.count_tags_words()

    def count_tags_words(self):
        tags = Counter()
        tag_pairs = defaultdict(dict)
        word_tag_pairs = defaultdict(dict)
        for sentence in self.train:
            for index in range(len(sentence) - 1):
                tag = sentence[index][1]
                tag_next = sentence[index + 1][1]
                if tag not in tag_pairs[tag_next] or tag_next not in tag_pairs:
                    tag_pairs[tag_next][tag] = 1
                else:
                    tag_pairs[tag_next][tag] += 1
            for word, tag in sentence:
                tags[tag] += 1
                if tag not in word_tag_pairs[word] or word not in word_tag_pairs:
                    word_tag_pairs[word][tag] = 1
                else:
                    word_tag_pairs[word][tag] += 1
        
        for tag in tags:
            word_tag_pairs['#UNKNOWN'][tag] = 0
        return tags, tag_pairs, word_tag_pairs


    def get_smoothed_probabilities(self, pairs1, pairs2):
        return_pairs = defaultdict(dict)
        for pair in pairs1:
            for tag in self.tags:
                temp = pairs2.get(pair, {}).get(tag, 0)
                return_pairs[pair][tag] = log(
                    (temp + self.laplace) / (self.tags[tag] + self.laplace * len(pairs1)))
        return return_pairs

    def construct_trellis(self):
        result = []
        for sentence in self.test:
            trellis, trellis_path = defaultdict(dict), defaultdict(dict)
            # initialization 
            for tag in self.tags:
                trellis[0][tag] = float("-inf")
                trellis_path[0][tag] = None
                if tag == "START":
                    trellis[0][tag] = 1
            # recurrent 
            for index in range(1, len(sentence)):
                word=sentence[index]
                for tag in self.tags:
                    max_path_p, max_path_previous_tag = None, None
                    for previous_tag in self.tags:
                        if word in self.emission and tag in self.emission[word]:
                            emission = self.emission[word][tag]
                        else: 
                            emission = self.emission['#UNKNOWN'][tag]
                        transition=self.transition[tag][previous_tag]
                        last_node_p=trellis[index - 1][previous_tag]
                        current_node_p = emission + transition+last_node_p
                        if max_path_p is None or current_node_p > max_path_p:
                            max_path_p = current_node_p
                            max_path_previous_tag = previous_tag
                    trellis[index][tag] = max_path_p
                    trellis_path[index][tag] = max_path_previous_tag
            temp = []
            index = len(sentence) - 1
            max_value = float("-inf")
            max_key = None
            for key in trellis[index].keys():
                if max_value < trellis[index][key]:
                    max_value = trellis[index][key]
                    max_key = key
            while max_key:
                temp.append((sentence[index], max_key))
                max_key = trellis_path[index][max_key]
                index -= 1
            result.append(temp[::-1])
        return result

    def hmm_train(self):
        self.emission = self.get_smoothed_probabilities(
            self.word_tag_pairs, self.word_tag_pairs)
        self.transition = self.get_smoothed_probabilities(
            self.tags, self.tag_pairs)

        self.prediction = self.construct_trellis()
  


def viterbi_1(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    hmm = HMM(train, test)
    hmm.hmm_train()

    return hmm.prediction

