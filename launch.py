# coding: utf-8

import argparse
from configparser import ConfigParser

from embed_rank.embeddings.emb_distrib_local import EmbeddingDistributorLocal
from embed_rank.model.input_representation import InputTextObj
from embed_rank.model.method import MMRPhrase
from embed_rank.preprocessing.postagging import PosTagging
from embed_rank.util.fileIO import read_file


def extract_keyphrases(emdist, ptagger, raw_text, N, lang, beta=0.55, alias_threshold=0.7):
    '''Method that extract a set of keyphrases.

    Args:
        emdist (EmbeddingDistributor)
        ptagger (PosTagger)
        raw_text (str): A string containing the raw text to extract.
        N (int): The number of keyphrases to extract.
        lang (str): Source text language.
        beta (float, optional): Beta factor for MMR (tradeoff informativness/diversity)
        alias_threshold (float, optional): Threshold to group candidates as aliases.

    Returns:
         A tuple with 3 elements :
            1)list of the top-N candidates (or less if there are not enough candidates) (list of string)
            2)list of associated relevance scores (list of float)
            3)list containing for each keyphrase a list of alias (list of list of string)
    '''
    tagged = ptagger.pos_tag_raw_text(raw_text)
    text_obj = InputTextObj(tagged, lang)
    return MMRPhrase(emdist, text_obj, N=N, beta=beta, alias_threshold=alias_threshold)


def main():
    '''Parse args and extract key phrases.
    '''
    parser = argparse.ArgumentParser(description='Extract keyphrases from raw text')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-raw_text', help='raw text to process')
    group.add_argument('-text_file', help='file containing the raw text to process')

    parser.add_argument('-N', help='number of keyphrases to extract', required=True, type=int)
    args = parser.parse_args()

    config = ConfigParser()
    config.read('config.ini')

    if args.text_file:
        raw_text = read_file(args.text_file)
    else:
        raw_text = args.raw_text

    sent2vec_model = config.get('SENT2VEC', 'model_path')
    print(f'Loading sent2vec model from {sent2vec_model}')
    embedding_distributor = EmbeddingDistributorLocal(sent2vec_model)

    spacy_model = config.get('SPACY', 'model')
    print(f'Loading spacy model {spacy_model}')
    pos_tagger = PosTagging(model=spacy_model)

    print(f'Extracting {args.N} keyphrases')
    print(extract_keyphrases(embedding_distributor, pos_tagger, raw_text, args.N, 'en'))


if __name__ == '__main__':
    main()