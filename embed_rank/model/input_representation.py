# Copyright (c) 2017-present, Swisscom (Schweiz) AG.
# All rights reserved.
#
#Authors: Kamil Bennani-Smires, Yann Savary

from nltk.stem import PorterStemmer


class InputTextObj:
    '''Represent the input text from which we want to extract keyphrases.
    '''

    def __init__(self, pos_tagged, stem=False, min_word_len=3):
        '''
        Args:
            pos_tagged (list): List of list : Text pos_tagged as a list of sentences
                where each sentence is a list of tuple (word, TAG).
            stem (bool, optional): If we want to apply stemming on the text.
            min_word_len (int, optional)
        '''
        self.min_word_len = min_word_len
        self.considered_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ'}
        self.pos_tagged = []
        self.filtered_pos_tagged = []

        if stem:
            stemmer = PorterStemmer()
            self.pos_tagged = [[(stemmer.stem(t[0]), t[1]) for t in sent] for sent in pos_tagged]
        else:
            self.pos_tagged = [[(t[0].lower(), t[1]) for t in sent] for sent in pos_tagged]

        temp = []
        for sent in self.pos_tagged:
            s = []
            for elem in sent:
                if len(elem[0]) < min_word_len:
                    s.append((elem[0], 'LESS'))
                else:
                    s.append(elem)
            temp.append(s)

        self.pos_tagged = temp
        self.filtered_pos_tagged = [[(t[0].lower(), t[1]) for t in sent if self.is_candidate(t)] for sent in
                                    self.pos_tagged]

    def is_candidate(self, tagged_token):
        '''

        Args:
            tagged_token (tuple): (word, tag)
        Returns:
            True if the passed token is a valid candidate word.
        '''
        return tagged_token[1] in self.considered_tags

    def extract_candidates(self):
        '''
        Returns:
            set: All candidate words.
        '''
        return {tagged_token[0].lower()
                for sentence in self.pos_tagged
                for tagged_token in sentence
                if self.is_candidate(tagged_token) and len(tagged_token[0]) >= self.min_word_len
                }