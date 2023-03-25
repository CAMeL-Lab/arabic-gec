import json
import csv
import os
import copy
import numpy as np
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.disambig.mle import MLEDisambiguator
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InputExample:
    """Simple object to encapsulate each data example"""
    def __init__(self, src_token, trg_token,
                 ged_tag):
        self.src_token = src_token
        self.trg_token = trg_token
        self.ged_tag = ged_tag

    def __repr__(self):
        return str(self.to_json_str())

    def to_json_str(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

class RawDataset:
    """Encapsulates the raw examples in InputExample objects"""
    def __init__(self, data_dir, first_person_only=False):
        self.train_examples = self.get_train_examples(data_dir)

        self.dev_examples = self.get_dev_examples(data_dir)

        self.test_examples = self.get_test_examples(data_dir)

    def create_examples(self, src_path, trg_path, ged_path):

        src_tokens = self.get_token_examples(src_path)
        trg_tokens = self.get_token_examples(trg_path)
        ged_tags = self.get_ged_tags(ged_path)

        examples = []

        for i in range(len(src_tokens)):
            src_token = src_tokens[i].strip()
            trg_token = trg_tokens[i].strip()
            ged_tag = ged_tags[i].strip()

            if self.is_model_tag(ged_tag):
                input_example = InputExample(src_token=src_token,
                                            trg_token=trg_token,
                                            ged_tag=ged_tag)

                examples.append(input_example)

        return examples

    def is_model_tag(self, tag):
        tag_combs = [
            'REPLACE_OH+REPLACE_OM',
            'REPLACE_OH+REPLACE_OT',
            'REPLACE_OD+REPLACE_OR',
            'REPLACE_OD+REPLACE_OG',
            'REPLACE_XC+REPLACE_XN',
            'REPLACE_OA+REPLACE_OH',
            'REPLACE_OM+REPLACE_OR',
            'REPLACE_OH+REPLACE_XC',
            'REPLACE_OD+REPLACE_OH',
            'REPLACE_XC+REPLACE_XG',
            'REPLACE_MI+REPLACE_OH',
            'REPLACE_OA+REPLACE_OR',
            'REPLACE_OR+REPLACE_OT',
            'REPLACE_OD+REPLACE_OM'
            ]

        if tag == 'UC' or tag == 'MERGE' or tag == 'UNK':
            return False

        if '+' in tag and tag not in tag_combs:
            return False

        return True

    def get_ged_tags(self, data_dir):
        with open(data_dir) as f:
            return f.readlines()

    def get_token_examples(self, data_dir):
        with open(data_dir, encoding='utf8') as f:
            return f.readlines()

    def get_train_examples(self, data_dir):
        """Reads the train examples of the dataset"""

        return self.create_examples(os.path.join(data_dir, 'nn_token_data/train.src.txt'),
                                    os.path.join(data_dir, 'nn_token_data/train.tgt.txt'),
                                    os.path.join(data_dir, 'nn_token_data/train.tags.txt'))


    def get_dev_examples(self, data_dir):
        """Reads the dev examples of the dataset"""
        return self.create_examples(os.path.join(data_dir, 'nn_token_data/tune.src.txt'),
                                    os.path.join(data_dir, 'nn_token_data/tune.tgt.txt'),
                                    os.path.join(data_dir, 'nn_token_data/tune.tags.txt'))


    def get_test_examples(self, data_dir):
        """Reads the test examples of the dataset"""
        return self.create_examples(os.path.join(data_dir, 'nn_token_data/test.src.txt'),
                                    os.path.join(data_dir, 'nn_token_data/test.tgt.txt'),
                                    os.path.join(data_dir, 'nn_token_data/test.tags.txt'))


class Vocabulary:
    """Base vocabulary class"""
    def __init__(self, token_to_idx=None):

        if token_to_idx is None:
            token_to_idx = dict()

        self.token_to_idx = token_to_idx
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}

    def add_token(self, token):
        if token in self.token_to_idx:
            index = self.token_to_idx[token]
        else:
            index = len(self.token_to_idx)
            self.token_to_idx[token] = index
            self.idx_to_token[index] = token
        return index

    def add_many(self, tokens):
        return [self.add_token(token) for token in tokens]

    def lookup_token(self, token):
        return self.token_to_idx[token]

    def lookup_index(self, index):
        return self.idx_to_token[index]

    def to_serializable(self):
        return {'token_to_idx': self.token_to_idx}

    @classmethod
    def from_serializable(cls, contents):
        return cls(**contents)

    def __len__(self):
        return len(self.token_to_idx)


class SeqVocabulary(Vocabulary):
    """Sequence vocabulary class"""
    def __init__(self, token_to_idx=None, unk_token='<unk>',
                 pad_token='<pad>', sos_token='<s>',
                 eos_token='</s>'):

        super(SeqVocabulary, self).__init__(token_to_idx)

        self.pad_token = pad_token
        self.unk_token = unk_token
        self.sos_token = sos_token
        self.eos_token = eos_token

        self.pad_idx = self.add_token(self.pad_token)
        self.unk_idx = self.add_token(self.unk_token)
        self.sos_idx = self.add_token(self.sos_token)
        self.eos_idx = self.add_token(self.eos_token)

    def to_serializable(self):
        contents = super(SeqVocabulary, self).to_serializable()
        contents.update({'unk_token': self.unk_token,
                         'pad_token': self.pad_token,
                         'sos_token': self.sos_token,
                         'eos_token': self.eos_token})

        return contents

    @classmethod
    def from_serializable(cls, contents):
        return cls(**contents)

    def lookup_token(self, token):
        return self.token_to_idx.get(token, self.unk_idx)


class Vectorizer:
    """Vectorizer Class"""
    def __init__(self, src_vocab_char, trg_vocab_char,
                 src_vocab_word, ged_tag_vocab,
                 add_side_constraints=False):
        """
        Args:
            - src_vocab_char (SeqVocabulary): source vocab on the char level
            - trg_vocab_char (SeqVocabulary): target vocab on the char level
            - src_vocab_word (SeqVocabulary): source vocab on the word level
            - ged_tag_vocab (Vocabulary): ged tags vocab on the word level
        """

        self.src_vocab_char = src_vocab_char
        self.trg_vocab_char = trg_vocab_char
        self.src_vocab_word = src_vocab_word
        self.ged_tag_vocab = ged_tag_vocab
        self.add_side_constraints = add_side_constraints

    @classmethod
    def create_vectorizer(cls, data_examples,
                          add_side_constraints=False):
        """Class method which builds the vectorizer
        vocab

        Args:
            - data_examples: list of InputExample

        Returns:
            - Vectorizer object
        """

        src_vocab_char = SeqVocabulary()
        trg_vocab_char = SeqVocabulary()
        src_vocab_word = SeqVocabulary()
        ged_tag_vocab = Vocabulary()

        for ex in data_examples:
            src_token = ex.src_token
            trg_token = ex.trg_token
            ged_tag = ex.ged_tag

            src_vocab_word.add_token(src_token)

            src_vocab_char.add_many(list(src_token))

            trg_vocab_char.add_many(list(trg_token))

            # adding ged tag to src and target vocab if needed
            if add_side_constraints:
                src_vocab_word.add_token(f'<{ged_tag}>')
                src_vocab_char.add_token(f'<{ged_tag}>')
                trg_vocab_char.add_token(f'<{ged_tag}>')
            
            # ged_tag is used for the non-side-constraints exps,
            # so we dont need <>
            ged_tag_vocab.add_token(ged_tag)

        logger.info(f"*** GED Tags:  {ged_tag_vocab.token_to_idx} ***")

        return cls(src_vocab_char, trg_vocab_char,
                   src_vocab_word, ged_tag_vocab,
                   add_side_constraints=add_side_constraints)

    def get_src_indices(self, seq, ged_tag=None):
        """Converts the source sequence chars
        to indices

        Args:
          - seq (str): The source sequence

        Returns:
          - char_level_indices (list): <s> + List of chars to index mapping + </s>
        """

        char_level_indices = [self.src_vocab_char.sos_idx]
        word_level_indices = [self.src_vocab_word.sos_idx]

        if self.add_side_constraints:
            char_level_indices.append(self.src_vocab_char.lookup_token(f'<{ged_tag}>'))
            word_level_indices.append(self.src_vocab_word.lookup_token(f'<{ged_tag}>'))

        for char in seq:
            char_level_indices.append(self.src_vocab_char.lookup_token(char))
            word_level_indices.append(self.src_vocab_word.lookup_token(seq))

        word_level_indices.append(self.src_vocab_word.eos_idx)
        char_level_indices.append(self.src_vocab_char.eos_idx)

        assert len(word_level_indices) == len(char_level_indices)

        return char_level_indices, word_level_indices

    def get_trg_indices(self, seq):
        """Converts the target sequence chars
        to indices

        Args:
          - seq (str): The target sequence

        Returns:
          - trg_x_indices (list): <s> + List of chars to index mapping
          - trg_y_indices (list): List of chars to index mapping + </s>
        """
        indices = [self.trg_vocab_char.lookup_token(t) for t in seq]

        trg_x_indices = [self.trg_vocab_char.sos_idx] + indices
        trg_y_indices = indices + [self.trg_vocab_char.eos_idx]
        return trg_x_indices, trg_y_indices

    def vectorize(self, src, trg, ged_tag):
        """
        Args:
          - src (str): The source sequence
          - trg (str): The target sequence
          - ged_tag (str): The target sequence ged

        Returns:
          - vectorized_src_char (tensor): <s> + vectorized source seq on the char level + </s>
          - vectorized_src_word (tensor): <s> + vectorized source seq on the word level + </s>
          - vectorized_trg_x (tensor): <s> + vectorized target seq on the char level
          - vectorized_trg_y (tensor): vectorized target seq on the char level + </s>
          - vectorized_ged_tag (tensor): vectorized ged tag
        """

        vectorized_src_char, vectorized_src_word = self.get_src_indices(src, ged_tag)
        vectorized_trg_x, vectorized_trg_y = self.get_trg_indices(trg)
        vectorized_ged_tag = self.ged_tag_vocab.lookup_token(ged_tag)

        return {'src_char': torch.tensor(vectorized_src_char, dtype=torch.long),
                'src_word': torch.tensor(vectorized_src_word, dtype=torch.long),
                'trg_x': torch.tensor(vectorized_trg_x, dtype=torch.long),
                'trg_y': torch.tensor(vectorized_trg_y, dtype=torch.long),
                'ged_tag': torch.tensor(vectorized_ged_tag, dtype=torch.long)
               }

    def to_serializable(self):
        return {'src_vocab_char': self.src_vocab_char.to_serializable(),
                'trg_vocab_char': self.trg_vocab_char.to_serializable(),
                'src_vocab_word': self.src_vocab_word.to_serializable(),
                'ged_tag_vocab': self.ged_tag_vocab.to_serializable()
               }

    @classmethod
    def from_serializable(cls, contents):
        src_vocab_char = SeqVocabulary.from_serializable(contents['src_vocab_char'])
        src_vocab_word = SeqVocabulary.from_serializable(contents['src_vocab_word'])
        trg_vocab_char = SeqVocabulary.from_serializable(contents['trg_vocab_char'])
        ged_tag_vocab = Vocabulary.from_serializable(contents['ged_tag_vocab'])

        return cls(src_vocab_char, trg_vocab_char,
                   src_vocab_word, ged_tag_vocab)


if __name__ == '__main__':
    dataset = RawDataset('/scratch/ba63/gec/data/alignment/modeling_areta_tags_improved/qalb14')
    vectorizer = Vectorizer.create_vectorizer(dataset.train_examples, add_side_constraints=True)
    import pdb; pdb.set_trace()
    x = 10