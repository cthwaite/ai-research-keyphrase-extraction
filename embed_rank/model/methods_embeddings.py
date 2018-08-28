# Copyright (c) 2017-present, Swisscom (Schweiz) AG.
# All rights reserved.
#
#Authors: Kamil Bennani-Smires, Yann Savary

import numpy as np

from .extractor import extract_candidates, extract_sent_candidates


def extract_doc_embedding(embedding_distrib, inp_rpr, use_filtered=False):
    '''Return the embedding of the full document.

    Args:
        embedding_distrib: embedding distributor see @EmbeddingDistributor
        inp_rpr: input text representation see @InputTextObj
        use_filtered: if true keep only candidate words in the raw text before computing the embedding

    Returns:
        numpy array of shape (1, dimension of embeddings) that contains the document embedding
    '''
    if use_filtered:
        tagged = inp_rpr.filtered_pos_tagged
    else:
        tagged = inp_rpr.pos_tagged

    tokenized_doc_text = ' '.join(token[0].lower() for sent in tagged for token in sent)
    return embedding_distrib.get_tokenized_sents_embeddings([tokenized_doc_text])


def extract_candidates_embedding_for_doc(embedding_distrib, inp_rpr):
    '''Return the list of candidate phrases as well as the associated numpy
    array that contains their embeddings.

    Note that candidate phrases extracted by PosTag rules which are unknown
    (in term of embeddings) will be removed from the candidates.

    Args:
        embedding_distrib: embedding distributor see @EmbeddingDistributor
        inp_rpr: input text representation see @InputTextObj

    Returns:
        A tuple of two element containing
            1) the list of candidate phrases
            2) a numpy array of shape (number of candidate phrases, dimension
                of embeddings: each row is the embedding of one candidate phrase
    '''
    candidates = np.array(extract_candidates(inp_rpr))  # List of candidates based on PosTag rules
    if len(candidates) > 0:
        embeddings = np.array(embedding_distrib.get_tokenized_sents_embeddings(candidates))  # Associated embeddings
        valid_candidates_mask = ~np.all(embeddings == 0, axis=1)  # Only candidates which are not unknown.
        return candidates[valid_candidates_mask], embeddings[valid_candidates_mask, :]
    else:
        return np.array([]), np.array([])


def extract_sent_candidates_embedding_for_doc(embedding_distrib, inp_rpr):
    '''Returns the list of candidate senetences as well as the associated numpy
    array that contains their embeddings.
    
    Note that candidate sentences which are unknown (in term of embeddings)
    will be removed from the candidates.

    Args:
        embedding_distrib: embedding distributor see @EmbeddingDistributor
        inp_rpr: input text representation see @InputTextObj

    Returns:
        A tuple of two element containing
            1) The list of candidate sentences
            2) A numpy array of shape (candidate sentences, embedding dimension);
    each row is the embedding of one candidate sentence
    '''
    candidates = np.array(extract_sent_candidates(inp_rpr))
    embeddings = np.array(embedding_distrib.get_tokenized_sents_embeddings(candidates))

    valid_candidates_mask = ~np.all(embeddings == 0, axis=1)
    return candidates[valid_candidates_mask], embeddings[valid_candidates_mask, :]
