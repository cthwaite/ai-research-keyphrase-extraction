# Copyright (c) 2017-present, Swisscom (Schweiz) AG.
# All rights reserved.
#
#Authors: Kamil Bennani-Smires, Yann Savary
'''Contain method that return list of candidate.
'''

import re

import nltk

GRAMMAR_EN = '''  NP:
        {<NN.*|JJ>*<NN.*>}  # Adjective(s)(optional) + Noun(s)'''


def extract_candidates(text_obj, no_subset=False):
    '''Based on part of speech return a list of candidate phrases
    
    Args:
        text_obj: Input text Representation see @InputTextObj
        lang (str) : language (currently en, fr and de are supported)
        no_subset (bool, Optional): If True won't put a candidate which is the
            subset of an other candidate

    Returns:
        string: list of candidate phrases
    '''

    keyphrase_candidate = set()

    np_parser = nltk.RegexpParser(GRAMMAR_EN)  # Noun phrase parser
    trees = np_parser.parse_sents(text_obj.pos_tagged)  # Generator with one tree per sentence

    for tree in trees:
        for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):  # For each nounphrase
            # Concatenate the token with a space
            keyphrase_candidate.add(' '.join(word for word, tag in subtree.leaves()))

    keyphrase_candidate = {kp for kp in keyphrase_candidate if len(kp.split()) <= 5}

    if no_subset:
        keyphrase_candidate = unique_ngram_candidates(keyphrase_candidate)
    else:
        keyphrase_candidate = list(keyphrase_candidate)

    return keyphrase_candidate


def extract_sent_candidates(text_obj):
    '''

    Args:
        text_obj: input Text Representation see @InputTextObj
    Returns:
        list: List of tokenized sentence (string), each token is separated by
            a space in the string.
    '''
    return [(' '.join(word for word, tag in sent)) for sent in text_obj.pos_tagged]


def unique_ngram_candidates(strings):
    '''
    ['machine learning', 'machine', 'backward induction', 'induction', 'start'] ->
    ['backward induction', 'start', 'machine learning']
    :param strings: List of string
    :return: List of string where no string is fully contained inside another string
    '''
    results = []
    for s in sorted(set(strings), key=len, reverse=True):
        if not any(re.search(r'\b{}\b'.format(re.escape(s)), r) for r in results):
            results.append(s)
    return results
