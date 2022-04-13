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
from collections import Counter
import time
# import pandas as pd


class HMM:
    def __init__(self, train, test) -> None:
        self.train = train
        self.test = test
        # all tags except 'START' and 'END'
        self.tags, self.words = self.get_tags_words()
        self.tag_freq = dict(
            Counter([tag for _, tag in self.flatten(self.train)]))
        self.n_tags, self.n_words, self.n_train = len(
            self.tags), len(self.words), len(self.train)
        self.emission = self.ini_nested_dict(
            self.tags, self.words+['#UNKNOWN'])
        # self.transition: added a 'START' tag for the beginning, to the tags.
        self.transition = self.ini_nested_dict(self.tags+['START'], self.tags)
        self.prediction = []
        self.laplace = 1e-5
        # cProfile.runctx('self.viterbi_forword()', globals(), locals())

    def flatten(self, list):
        return [item for sublist in list for item in sublist]

    def get_tags_words(self):
        tags = list(set([tag for _, tag in self.flatten(self.train)]))
        tags.remove('START')
        tags.remove('END')
        words = list(set([word for word, _ in self.flatten(self.train)]))
        return tags, words

    def ini_nested_dict(self, list1, list2):
        return {item1: {item2: 0 for item2 in list2} for item1 in list1}

    def unknown_word_emission(self, tag, word):
        try:
            return self.emission[tag][word]
        except:
            return self.emission[tag]['#UNKNOWN']

    def viterbi_forword(self):
        # pi is initial
        # a is transition
        # b is emission
        # x_t,o_t is t-th order word/node
        # i,j is previous tag/state, current tag/state
        # s is tag/state

        initial = self.transition["START"]
        # (pd.DataFrame.from_dict(self.emission)==0).sum()
        for i, sentence in enumerate(self.test):
            if i % 1000 == 0:
                print(i)
            # if i > 100:
            #     break
            # recursion all words
            sentence = sentence[1:-1]
            viterbi = self.ini_nested_dict(sentence, self.tags)
            backpointer = self.ini_nested_dict(sentence, self.tags)

            for index, word in enumerate(sentence):
                for tag in self.tags:
                    # Initial Node Probability; initialization step
                    if index == 0:
                        viterbi[word][tag] = log(
                            initial[tag])+log(self.unknown_word_emission(tag, word))
                        backpointer[word][tag] = 'START'
                    # other node
                    else:
                        # Edge Probability
                        a = log([self.transition[tag_pre][tag]
                                for tag_pre in self.tags])
                        b = log(
                            self.unknown_word_emission(tag, word))
                        # Node Probability
                        last_nodes = np.array(
                            list(viterbi[sentence[index-1]].values()))
                        last_viterbi_transition=last_nodes+a
                        last_viterbi_transition_max = max(last_viterbi_transition)
                        last_viterbi_transition_max_which=self.tags[np.argmax(last_viterbi_transition)]
                        ## calc this node Probability
                        viterbi[word
                                ][tag] = last_viterbi_transition_max+b
                        # Backpointer
                        backpointer[word][tag] = last_viterbi_transition_max_which


                        # # Node Probability
                        # possible_viterbi = a+b + \
                        #     list(viterbi[sentence[index-1]].values())
                        # viterbi[word][tag] = max(possible_viterbi)

            self.viterbi_predict(viterbi, backpointer, sentence)



    def viterbi_predict(self, viterbi, backpointer, sentence):
        # terminationstep
        possible_viterbi = list(viterbi[sentence[-1]].values())
        # bestpathprob = max(possible_viterbi)
        bestpathpointer = self.tags[np.argmax(possible_viterbi)]

        # get best path
        bestpath = [bestpathpointer]
        last_tag = bestpathpointer
        for word in reversed(sentence[1:]):
            current_tag = backpointer[word][last_tag]
            bestpath.insert(0, current_tag)
            last_tag = current_tag

        # merge word and predicted POS
        pred=[(word, tag) for word, tag in zip(sentence, bestpath)]
        pred.insert(0,("START","START"))
        pred.append(('END','END'))
        self.prediction.append(pred)

    def laplace_sm(self, numerator_count, denominator_count, corpus_len, laplace):
        return (numerator_count + laplace) / (denominator_count +
                                                   laplace * corpus_len)

    def get_initial(self):
        # initial prob
        # initialization
        initial_ct = self.transition['START']

        # count numerator
        for sentence in self.train:
            ini_word_tag = sentence[1][1]
            initial_ct[ini_word_tag] += 1

        # laplace smoothing
        for ini_word_tag in self.tags:
            self.transition['START'][ini_word_tag] = self.laplace_sm(
                initial_ct[ini_word_tag], self.n_train, self.n_tags, self.laplace)
        pass

    def get_transition(self):
        # transition prob
        # initialization
        transition_ct = self.transition

        # count numerator
        for sentence in self.train:
            for index in range(2,len(sentence)-1):
                pre_word_tag = sentence[index-1][1]
                cur_word_tag = sentence[index][1]
                transition_ct[pre_word_tag][cur_word_tag] += 1

        # laplace smoothing
        for tag_pre in self.tags:
            for tag_cur in self.tags:
                self.transition[tag_pre][tag_cur] = self.laplace_sm(
                    transition_ct[tag_pre][tag_cur], self.tag_freq[tag_pre], self.n_tags, self.laplace)
        pass

    def get_emission(self):
        # emission prob
        # initialization
        emission_ct = self.emission

        # count numerator
        for sentence in self.train:
            for index in range(1,len(sentence)-1):
                word = sentence[index][0]
                tag = sentence[index][1]
                emission_ct[tag][word] += 1

        # laplace smoothing
        for tag in self.tags:
            for word in self.words+['#UNKNOWN']:
                # if word=='#UNKNOWN':
                #     print(word)
                self.emission[tag][word] = self.laplace_sm(
                    emission_ct[tag][word], self.tag_freq[tag], self.n_words+1, self.laplace)

        pass

    def train_hmm(self):
        self.get_initial()
        self.get_transition()
        self.get_emission()



def viterbi_1(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    hmm = HMM(train, test)
    hmm.train_hmm()
    hmm.viterbi_forword()

    # debug
    ct=Counter(hmm.flatten(hmm.prediction))
    {(word,tag):count for (word,tag),count in ct.items() if word==','}
    {tag:hmm.emission[tag][','] for tag in hmm.tags}

    [sentence for sentence in hmm.prediction if (',', 'CONJ') in sentence]

    return hmm.prediction
