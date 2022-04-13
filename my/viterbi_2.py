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
from collections import Counter
from viterbi_1 import HMM

class HMM2(HMM):
    def __init__(self, train, test):
        super(HMM2, self).__init__(train, test)
        self.hapax=self.get_hapax()
    
    def get_hapax(self):
        # word_ct=Counter([word for word, _ in self.flatten(self.train)])
        word_ct = Counter(self.flatten(self.train))
        hapax = [word for word, count in word_ct.items() if count == 1]
        tagfreq = Counter([tag for _,tag in hapax])

        unseen=list(set(self.tags)-set(tagfreq.keys()))
        for tag in unseen:
            tagfreq[tag]=1

        return {tag: count/len(hapax) for tag, count in tagfreq.items()}
        

    def get_emission(self):
        # emission prob
        # initialization
        emission_ct = self.emission

        # count numerator
        for sentence in self.train:
            for index in range(1, len(sentence)-1):
                word = sentence[index][0]
                tag = sentence[index][1]
                emission_ct[tag][word] += 1

        # laplace smoothing
        for tag in self.tags:
            for word in self.words+['#UNKNOWN']:
                # if word=='#UNKNOWN':
                #     print(word)
                self.emission[tag][word] = self.laplace_sm(
                    emission_ct[tag][word], self.tag_freq[tag], self.n_words+1, self.hapax[tag]*self.laplace)

        pass


def viterbi_2(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    hmm = HMM2(train, test)
    hmm.train_hmm()
    hmm.viterbi_forword()

    return hmm.prediction
