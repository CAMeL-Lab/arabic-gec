# -*- coding: utf-8 -*-

# MIT License
#
# Copyright 2021 New York University Abu Dhabi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

""" Token classification fine-tuning: utilities to work with token
    classification tasks (NER, POS tagging, etc.)
    Heavily adapted from: https://github.com/huggingface/transformers/blob/
    v3.0.1/examples/token-classification/utils_ner.py"""


import logging
import os
from dataclasses import dataclass
from filelock import FileLock
from enum import Enum
from typing import List, Optional, Union

import torch
import torch.nn as nn
import numpy as np
from datasets import Dataset
from transformers import PreTrainedTokenizer


logger = logging.getLogger(__name__)


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"
    tune = "tune"


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

    tokenized_inputs = tokenizer(examples_tokens, truncation=True, is_split_into_words=True)

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



