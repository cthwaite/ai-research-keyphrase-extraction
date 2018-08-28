# Copyright (c) 2017-present, Swisscom (Schweiz) AG.
# All rights reserved.
#
#Authors: Kamil Bennani-Smires, Yann Savary

import argparse
import os
import re
import warnings

import spacy

from ..util.fileIO import read_file, write_string


class PosTagging:
    '''Parts-of-speech tagging using spaCy.
    '''
    def __init__(self, nlp=None, model='en_core_web_sm'):
        '''
        Args:
            nlp
            model (str): spaCy model to load.
        '''
        if not nlp:
            print('Loading Spacy model')
            self.nlp = spacy.load(model, entity=False)
            print(f'Spacy model loaded: {model}')
        else:
            self.nlp = nlp

    def pos_tag_raw_text(self, text, as_tuple_list=True):
        '''Tokenize and POS tag a string.

        Sentence level is kept in the result :
        Either we have a list of list (for each sentence a list of tuple (word,tag))
        Or a separator [ENDSENT] if we are requesting a string by putting as_tuple_list = False

        Example :
        >>from sentkp.preprocessing import postagger as pt

        >>pt = postagger.PosTagger()

        >>pt.pos_tag_raw_text('Write your python code in a .py file. Thank you.')
        [
            [('Write', 'VB'), ('your', 'PRP$'), ('python', 'NN'),
            ('code', 'NN'), ('in', 'IN'), ('a', 'DT'), ('.', '.'), ('py', 'NN'), ('file', 'NN'), ('.', '.')
            ],
            [('Thank', 'VB'), ('you', 'PRP'), ('.', '.')]
        ]

        >>pt.pos_tag_raw_text('Write your python code in a .py file. Thank you.', as_tuple_list=False)

        'Write/VB your/PRP$ python/NN code/NN in/IN a/DT ./.[ENDSENT]py/NN file/NN ./.[ENDSENT]Thank/VB you/PRP ./.'

        >>pt = postagger.PosTagger(separator='_')
        >>pt.pos_tag_raw_text('Write your python code in a .py file. Thank you.', as_tuple_list=False)
        Write_VB your_PRP$ python_NN code_NN in_IN a_DT ._. py_NN file_NN ._.
        Thank_VB you_PRP ._.

        Args:
            text (str): String to POS tag.
            as_tuple_list (bool, optional): Return result as list of list (word,Pos_tag)
        
        Returns:
            POS Tagged string or tuple list.
        '''

        # This step is not necessary in the stanford tokenizer.
        # This is used to avoid such tags :  ('      ', 'SP')
        text = re.sub('[ ]+', ' ', text).strip()  # Convert multiple whitespaces into one

        doc = self.nlp(text)
        if as_tuple_list:
            return [[(token.text, token.tag_) for token in sent] for sent in doc.sents]
        return '[ENDSENT]'.join(' '.join('|'.join([token.text, token.tag_]) for token in sent) for sent in doc.sents)


    def pos_tag_file(self, input_path, output_path=None):
        '''Tokenize and POS tag a file.

        Note: The jumpline is only for readibility purposes; when reading a
        tagged file we use `sent_tokenize` to find the sentence boundaries.

        Args:
            input_path (str): path of the source file
            output_path (str, optional): If set, write POS tagged text with
                separators. If not set, return list of list of tuples.

        Returns:
            list: Resulting POS tagged text as a list of list of tuple.
            None: If `output_path` is set.
        '''

        original_text = read_file(input_path)

        if output_path is None:
            return self.pos_tag_raw_text(original_text, as_tuple_list=True)
    
        tagged_text = self.pos_tag_raw_text(original_text, as_tuple_list=False)
        # Write to the output the POS-Tagged text.
        write_string(tagged_text, output_path)

    def pos_tag_and_write_corpora(self, list_of_path, suffix):
        '''POS tag a list of files.

        It writes the resulting file in the same directory with the same name + suffix
        e.g
        pos_tag_and_write_corpora(['/Users/user1/text1', '/Users/user1/direct/text2'] , suffix = _POS)
        will create
        /Users/user1/text1_POS
        /Users/user1/direct/text2_POS

        Args:
            list_of_path (list): list containing the path (as string) of each
                file to POS tag.
            suffix (str): suffix to append at the end of the original filename
                for the resulting pos_tagged file.
        '''
        for path in list_of_path:
            output_file_path = path + suffix
            if os.path.isfile(path):
                self.pos_tag_file(path, output_file_path)
            else:
                warnings.warn(f'File {output_file_path} does not exist')



def main():
    '''Parse args and run tagger.
    '''
    parser = argparse.ArgumentParser(description='Write POS tagged files, the resulting file will be written'
                                     ' at the same location with _POS append at the end of the filename')
    parser.add_argument('-l', '--listing-file', help='Path to a text file '
                        'containing in each row a path to a file to POS tag')
    parser.add_argument('-p', '--phrases', help='Semicolon-separated list of'
                        'phrases to POS tag')
    args = parser.parse_args()

    if args.listing_file:
        tagger = PosTagging()
        list_of_path = read_file(args.listing_file).splitlines()

        print('POS Tagging and writing ', len(list_of_path), 'files')
        tagger.pos_tag_and_write_corpora(list_of_path, '_POS')
    else:
        tagger = PosTagging()
        for phrase in args.phrases.split(';'):
            data = tagger.pos_tag_raw_text(phrase)
            print(data)


if __name__ == '__main__':
    main()