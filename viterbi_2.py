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
Part 3: Here you should improve viterbi to use better laplace smoothing for unseen words
This should do better than baseline and your first implementation of viterbi, especially on unseen words
"""
from collections import Counter, defaultdict
from math import log
from viterbi_1 import HMM

class HMM2(HMM):
    def __init__(self, train, test):
        super(HMM2, self).__init__(train, test)
        self.hapax=self.get_hapax()
        self.hapax_possibility=self.hapax_distribution()


    def get_hapax(self):
        hapax = Counter()
        for sentence in self.train:
            for word, _ in sentence:
                hapax[word] += 1
        return hapax
    
    def hapax_distribution(self):
        hapax_tags = Counter()
        index = 0
        for hapax_word in self.hapax:
            if self.hapax[hapax_word] == 1:
                index += 1
                (key, _), = self.word_tag_pairs[hapax_word].items()
                hapax_tags[key] += 1
        return_pairs = Counter()
        for tag in self.tags:
            return_pairs[tag] = (hapax_tags[tag] + 1) / index
        return return_pairs

    def get_smoothed_probabilities_hapax(self,  pairs1, pairs2):
        return_pairs = defaultdict(dict)
        for pair in pairs1:
            for tag in self.tags:
                if pair == '#UNKNOWN':
                    laplace = self.hapax_possibility[tag]*self.laplace
                    # laplace = self.laplace
                else: 
                    laplace=self.laplace
                temp = pairs2.get(pair, {}).get(tag, 0)
                return_pairs[pair][tag] = log(
                    (temp + laplace) / (self.tags[tag] +laplace * len(pairs1)))
        return return_pairs
    
    def hmm_train(self):
        self.emission = self.get_smoothed_probabilities_hapax(
            self.word_tag_pairs, self.word_tag_pairs)
        self.transition = self.get_smoothed_probabilities(
            self.tags, self.tag_pairs)

        self.prediction = self.construct_trellis()
        
def viterbi_2(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    hmm = HMM2(train, test)
    hmm.hmm_train()

    return hmm.prediction
    
