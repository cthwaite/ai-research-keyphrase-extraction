# Copyright (c) 2017-present, Swisscom (Schweiz) AG.
# All rights reserved.
#
#Authors: Kamil Bennani-Smires, Yann Savary

from abc import ABC, abstractmethod


class EmbeddingDistributor(ABC):
    '''Abstract class in charge of providing the embeddings of piece of texts.
    '''
    @abstractmethod
    def get_tokenized_sents_embeddings(self, sents):
        '''Generate a numpy ndarray with the embedding of each element of sent in each row.

        Args:
            sents (list): list of string (sentences/phrases)

        Returns:
            ndarray: with shape (len(sents), dimension of embeddings)
        '''
        pass
