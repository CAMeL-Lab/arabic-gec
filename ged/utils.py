import logging
import os
from enum import Enum
from typing import List, Union

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer


logger = logging.getLogger(__name__)


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"
    test_L1 = "test_L1"
    test_L2 = "test_L2"


def read_examples_from_file(data_dir, mode: Union[Split, str]):
    if isinstance(mode, Split):
        mode = mode.value

    file_path = os.path.join(data_dir, f"{mode}.txt")
    guid_index = 1
    examples = {'words': [], 'labels': []}

    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples['words'].append(words)
                    examples['labels'].append(labels)
                    guid_index += 1
                    words = []
                    labels = []
            else:
                splits = line.split("\t")
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    # This is needed to get around the Trainer evaluation
                    labels.append("UC") # place holder
        if words:
            examples['words'].append(words)
            examples['labels'].append(labels)

    dataset = Dataset.from_dict(examples)
    return dataset


def process(examples, label_list: List[str], tokenizer: PreTrainedTokenizer):

    label_map = {label: i for i, label in enumerate(label_list)}

    examples_tokens = [words for words in examples['words']]

    tokenized_inputs = tokenizer(examples_tokens, is_split_into_words=True)

    labels = []
    examples_labels = [labels for labels in examples['labels']]

    for i, ex_labels in enumerate(examples_labels):

        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)

            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label = ex_labels[word_idx]
                if label == 'UNK':
                    label_ids.append(-200)
                else:
                    label_ids.append(label_map[label])

            else:
                label_ids.append(-100)

            previous_word_idx = word_idx


        assert len(label_ids) == len(word_ids)
        labels.append(label_ids)


    tokenized_inputs['labels'] = labels

    return tokenized_inputs


def get_labels(path: str) -> List[str]:
    with open(path, "r") as f:
        labels = f.read().splitlines()

    if 'UNK' in labels:
        labels.remove('UNK')

    return labels


class TokenClassificationDataset(torch.utils.data.Dataset):
    """A wrapper class for prediction dataset."""
    def __init__(self, examples, labels, tokenizer):
        self.tokenizer = tokenizer
        self.features = self.process_examples(examples, labels,
                                             pad_token_label_id=-100)

    def process_examples(self, examples, labels, pad_token_label_id=-100):
        label_map = {label: i for i, label in enumerate(labels)}

        examples_tokens = [words for words in examples['words']]

        examples_labels = [labels for labels in examples['labels']]

        featurized_inputs = []

        for ex_id, (example_tokens, example_labels) in enumerate(zip(examples_tokens, examples_labels)):
            tokens = []
            label_ids = []

            for word, label in zip(example_tokens, example_labels):
                word_tokens = self.tokenizer.tokenize(word)

                if len(word_tokens) > 0:
                    tokens.append(word_tokens)
                    if label == 'UNK':
                        label_ids.append([-200] +
                                        [pad_token_label_id] *
                                        (len(word_tokens) - 1))
                    else:
                        label_ids.append([label_map[label]] +
                                        [pad_token_label_id] *
                                        (len(word_tokens) - 1))

            token_segments = []
            token_segment = []
            label_ids_segments = []
            label_ids_segment = []
            num_word_pieces = 0
            seg_seq_length = self.tokenizer.model_max_length - 2

            for idx, word_pieces in enumerate(tokens):
                if num_word_pieces + len(word_pieces) > seg_seq_length:
                    # convert to ids and add special tokens

                    input_ids = self.tokenizer.convert_tokens_to_ids(token_segment)
                    input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]

                    label_ids_segment = [pad_token_label_id] + label_ids_segment + [pad_token_label_id]


                    features = {'input_ids': input_ids,
                                'attention_mask': [1] * len(input_ids),
                                'token_type_ids': [0] * len(input_ids),
                                'labels': label_ids_segment,
                                'sent_id': ex_id
                                }

                    featurized_inputs.append(features)

                    token_segments.append(token_segment)
                    label_ids_segments.append(label_ids_segment)
                    token_segment = list(word_pieces)
                    label_ids_segment = list(label_ids[idx])
                    num_word_pieces = len(word_pieces)
                else:
                    token_segment.extend(word_pieces)
                    label_ids_segment.extend(label_ids[idx])
                    num_word_pieces += len(word_pieces)

            if len(token_segment) > 0:
                input_ids = self.tokenizer.convert_tokens_to_ids(token_segment)
                input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]


                label_ids_segment = [pad_token_label_id] + label_ids_segment + [pad_token_label_id]

                features = {'input_ids': input_ids,
                            'attention_mask': [1] * len(input_ids),
                            'token_type_ids': [0] * len(input_ids),
                            'labels': label_ids_segment,
                            'sent_id': ex_id
                            }

                featurized_inputs.append(features)

                token_segments.append(token_segment)
                label_ids_segments.append(label_ids_segment)

        return featurized_inputs

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]
