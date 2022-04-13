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
This file should not be submitted - it is only meant to test your implementation of the Viterbi algorithm. 

See Piazza post @650 - This example is intended to show you that even though P("back" | RB) > P("back" | VB), 
the Viterbi algorithm correctly assigns the tag as VB in this context based on the entire sequence. 
"""
from utils import read_files, get_nested_dictionaries
from numpy import argmax, log
import numpy as np
import cProfile


def max_key(dict):
    return max((dict).keys(), key=lambda key: dict[key])

def main():
    # pi is initial
    # a is transition
    # b is emission
    # x_t,o_t is t-th order word/node
    # i,j is previous tag/state, current tag/state
    # s is tag/state
    test, emission, transition, output = read_files()
    emission, transition = get_nested_dictionaries(emission, transition)

    
    """WRITE YOUR VITERBI IMPLEMENTATION HERE"""
    initial = transition["START"]

    tags = list(transition.keys())
    tags.remove('START')
    for sentence in test:
        # recursion all words
        viterbi = {word: {tag: 0 for tag in tags} for word in sentence}
        backpointer = {word: {tag: 0 for tag in tags} for word in sentence}
        for index, word in enumerate(sentence):
            for tag in tags:
                # Initial Node Probability; initialization step
                if index==0:
                    viterbi[word][tag] = log(initial[tag])+log(emission[tag][word])
                    print(viterbi[word][tag])
                    backpointer[word][tag] = 'START'
                # other node 
                else:
                    # Edge Probability
                    a = log([transition[tag_pre][tag] for tag_pre in tags])
                    b = log(emission[tag][word])
                    # Node Probability
                    last_nodes = np.array(list(viterbi[sentence[index-1]].values()))
                    last_viterbi_transition=last_nodes+a
                    last_viterbi_transition_max = max(last_viterbi_transition)
                    last_viterbi_transition_max_which=tags[np.argmax(last_viterbi_transition)]
                    ## calc this node Probability
                    viterbi[word
                            ][tag] = last_viterbi_transition_max+b
                    # print(viterbi[word][tag])
                    # Backpointer
                    backpointer[word
                                ][tag] = last_viterbi_transition_max_which
        # terminationstep
        possible_viterbi = list(viterbi[sentence[-1]].values())
        bestpathprob = max(possible_viterbi)
        bestpathpointer = tags[np.argmax(possible_viterbi)]
        
        # get best path
        bestpath=[bestpathpointer]
        for word in reversed(sentence):
            last_tag=bestpathpointer
            current_tag=backpointer[word][last_tag]
            bestpath.insert(0,current_tag)
        bestpath.remove('START')
        
        # merge word and predicted POS
        prediction=[(word,tag) for word,tag in zip(sentence,bestpath)]



    print('Your Output is:',prediction,'\n Expected Output is:',output)


if __name__=="__main__":
    # cProfile.run('main()')
    main()