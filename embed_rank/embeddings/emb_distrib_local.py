# Copyright (c) 2017-present, Swisscom (Schweiz) AG.
# All rights reserved.
#
#Authors: Kamil Bennani-Smires, Yann Savary

import sent2vec

from .emb_distrib_interface import EmbeddingDistributor


class EmbeddingDistributorLocal(EmbeddingDistributor):
    '''Concrete class of @EmbeddingDistributor using a local installation of
    sent2vec.
    '''

    def __init__(self, fasttext_model):
        self.model = sent2vec.Sent2vecModel()
        self.model.load_model(fasttext_model)

    def get_tokenized_sents_embeddings(self, sents):
        '''@see EmbeddingDistributor
        '''
        for sent in sents:
            if '\n' in sent:
                raise RuntimeError('New line is not allowed inside a sentence')

        return self.model.embed_sentences(sents)
