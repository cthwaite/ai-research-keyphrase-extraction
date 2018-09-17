# coding: utf-8

import argparse
from configparser import ConfigParser

from embed_rank.embeddings.emb_distrib_local import EmbeddingDistributorLocal
from embed_rank.model.input_representation import InputTextObj
from embed_rank.model.method import MMRPhrase, MMRSent
from embed_rank.preprocessing.postagging import PosTagging
from embed_rank.util.fileIO import read_file


def extract_keyphrases(emdist, ptagger, raw_text, count, beta, alias_threshold, x_type='phrase'):
    '''Extract a set of keyphrases from a string.

    Args:
        emdist (EmbeddingDistributor)
        ptagger (PosTagger)
        raw_text (str): A string containing the raw text to extract.
        count (int): The number of keyphrases to extract.
        lang (str): Source text language.
        beta (float, optional): Beta factor for MMR, indicating the tradeoff
            between informativness and diversity.
        alias_threshold (float, optional): Threshold to group candidates as aliases.

    Returns:
         A tuple with 3 elements :
            1)list of the top-N candidates (or less if there are not enough candidates) (list of string)
            2)list of associated relevance scores (list of float)
            3)list containing for each keyphrase a list of alias (list of list of string)
    '''
    tagged = ptagger.pos_tag_raw_text(raw_text)
    text_obj = InputTextObj(tagged)
    
    if x_type == 'phrase':
        return MMRPhrase(emdist, text_obj, N=count, beta=beta, alias_threshold=alias_threshold)
    elif x_type == 'sentence':
        return MMRSent(emdist, text_obj, N=count, beta=beta, alias_threshold=alias_threshold)
    else:
        raise ValueError(f'Unknown feature type `{x_type}`')    


def main():
    '''Parse args and extract key phrases.
    '''
    parser = argparse.ArgumentParser(description='Extract keyphrases from raw text')

    group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument('-a', '--alias-threshold',
                        help='Threshold to group candidates as aliases',
                        default=0.7,
                        type=float)
    parser.add_argument('-b', '--beta',
                        help='Beta factor for MMR (tradeoff informativness/diversity)',
                        default=0.55,
                        type=float)
    parser.add_argument('-c', '--count',
                        help='Number of keyphrases to extract',
                        default=10,
                        type=int)
    group.add_argument('-r', '--raw-text',
                       help='Raw text to process')
    group.add_argument('-t', '--text-file',
                       help='File containing raw text to process')
    parser.add_argument('-x', '--x-type',
                        default='phrase',
                        choices=['phrase', 'sentence'],
                        help='Feature type to extract')
    args = parser.parse_args()

    if args.text_file:
        raw_text = read_file(args.text_file)
    else:
        raw_text = args.raw_text

    config = ConfigParser()
    config.read('config.ini')

    sent2vec_model = config.get('SENT2VEC', 'model_path')
    print(f'Loading sent2vec model from {sent2vec_model}')
    embedding_distributor = EmbeddingDistributorLocal(sent2vec_model)

    spacy_model = config.get('SPACY', 'model')
    print(f'Loading spacy model {spacy_model}')
    pos_tagger = PosTagging(model=spacy_model)

    print(f'Extracting {args.count} keyphrases')
    keyphrases = extract_keyphrases(embedding_distributor,
                                    pos_tagger,
                                    raw_text,
                                    args.count,
                                    args.beta,
                                    args.alias_threshold,
                                    args.x_type)
    print(keyphrases)


if __name__ == '__main__':
    main()