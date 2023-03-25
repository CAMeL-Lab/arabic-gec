import copy
import json
import torch
from torch.utils.data import Dataset


class InputExample:
    """Simple object to encapsulate each data example"""
    def __init__(self, word, tag, morph_feats):
        self.word = word
        self.tag = tag
        self.morph_feats = morph_feats

    def __repr__(self):
        return str(self.to_json_str())

    def to_json_str(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output


class RawDataset:
    def __init__(self, raw_data_dir):
        self.examples = self.create_examples(raw_data_dir)

    def create_examples(self, raw_data_dir):

        with open(raw_data_dir, mode='r') as f:
            examples = json.load(f)

        input_examples = []
        for token, tag, feats in zip(examples['tokens'], examples['tags'], examples['morph_feats']):
            input_examples.append(InputExample(word=token, tag=tag, morph_feats=feats))

        return input_examples


class Vectorizer:
    def __init__(self, char_vocab, error_tags_vocab):
        self.char_vocab = char_vocab
        self.error_tags_vocab = error_tags_vocab

    @classmethod
    def create_vectorizer(cls, dataset):
        char_vocab = CharVocab()
        error_tags_vocab = Vocabulary()

        for ex in dataset.examples:
            word = ex.word
            morph_feats = ex.morph_feats
            tag = ex.tag

            char_vocab.add_many(list(word))
            char_vocab.add_token(f'<{morph_feats}>')

            error_tags_vocab.add_token(tag)

        return cls(char_vocab, error_tags_vocab)

    def get_word_indices(self, word, morph_feats):
        vectorized_feats = [self.char_vocab.lookup_token(f'<{morph_feats}>')]
        vectorized_word = vectorized_feats + [self.char_vocab.lookup_token(c) for c in word]
        # vectorized_word = [self.char_vocab.lookup_token(c) for c in word]
        return torch.tensor([vectorized_word])

    def vectorize(self, word, tag, morph_feats):
        vectorized_feats = [self.char_vocab.lookup_token(f'<{morph_feats}>')]
        vectorized_word = vectorized_feats + [self.char_vocab.lookup_token(c) for c in word]
        # vectorized_word = [self.char_vocab.lookup_token(c) for c in word]
        vectorized_tag = [self.error_tags_vocab.lookup_token(tag)]

        return {'word': torch.LongTensor(vectorized_word),
                'tag': torch.LongTensor(vectorized_tag)
               }

    def to_serializable(self):
        return {'char_vocab': self.char_vocab.to_serializable(),
                'error_tags_vocab': self.error_tags_vocab.to_serializable()
                }

    @classmethod
    def from_serializable(cls, contents):
        char_vocab = CharVocab.from_serializable(contents['char_vocab'])
        error_tags_vocab = Vocabulary.from_serializable(contents['error_tags_vocab'])

        return cls(char_vocab, error_tags_vocab)


class Vocabulary:
    """Base vocabulary class"""
    def __init__(self, token_to_idx=None, unk_token='<unk>'):
        if token_to_idx is None:
            token_to_idx = dict()

        self.token_to_idx = token_to_idx
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
        self.unk_token = unk_token
        self.unk_idx = self.add_token(self.unk_token)

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
        return self.token_to_idx.get(token, self.unk_idx)

    def lookup_index(self, index):
        return self.idx_to_token[index]

    def to_serializable(self):
        return {'token_to_idx': self.token_to_idx}

    @classmethod
    def from_serializable(cls, contents):
        return cls(**contents)

    def __len__(self):
        return len(self.token_to_idx)


class CharVocab(Vocabulary):
    def __init__(self, token_to_idx=None, pad_token='<pad>'):
        super(CharVocab, self).__init__(token_to_idx)
        self.pad_token = pad_token
        self.pad_idx = self.add_token(self.pad_token)

    def to_serializable(self):
        contents = super(CharVocab, self).to_serializable()
        contents.update({'pad_token': self.pad_token})

        return contents

    @classmethod
    def from_serializable(cls, contents):
        return cls(**contents)

    def lookup_token(self, token):
        return self.token_to_idx.get(token, self.unk_idx)


class ErrorTagDataset(Dataset):
    def __init__(self, examples, vectorizer):
        self.examples = examples
        self.vectorizer = vectorizer

    @classmethod
    def load_data_and_create_vectorizer(cls, data_dir):
        raw_examples = RawDataset(data_dir)
        vectorizer = Vectorizer.create_vectorizer(raw_examples)
        return cls(raw_examples, vectorizer)

    @classmethod
    def load_data(cls, data_dir):
        raw_examples = RawDataset(data_dir)
        return cls(raw_examples, None)

    def __getitem__(self, idx):
        word = self.examples.examples[idx].word
        tag = self.examples.examples[idx].tag
        morph_feats = self.examples.examples[idx].morph_feats
        return self.vectorizer.vectorize(word, tag, morph_feats)

    def __len__(self):
        return len(self.examples.examples)

    def save_vectorizer(self, vec_path):
        with open(vec_path, 'w') as f:
            return json.dump(self.vectorizer.to_serializable(), f,
                             ensure_ascii=False)

    @staticmethod
    def load_vectorizer(vec_path):
        with open(vec_path) as f:
            return Vectorizer.from_serializable(json.load(f))

# if __name__ == '__main__':
#     train_dataset = ErrorTagDataset.load_data_and_create_vectorizer('/scratch/ba63/gec/data/alignment/modeling_areta_tags_check/'\
#                                     'qalb14/corruption_data/train.error_tagger.json')

#     # dev_dataset = ErrorTagDataset('/scratch/ba63/gec/data/alignment/modeling_areta_tags_check/'\
#     #                                 'qalb14/corruption_data/tune.tagger.txt')

#     import pdb; pdb.set_trace()
#     x = 10