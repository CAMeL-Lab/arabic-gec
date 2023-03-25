# -*- coding: utf-8 -*-

# MIT License
#
# Copyright 2018-2022 New York University Abu Dhabi
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

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertForTokenClassification, BertTokenizer
from enum import Enum
from typing import List, Optional, Union
import logging
import os
from dataclasses import dataclass
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

@dataclass
class InputExample:
    """
    A single training/test example for token classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word of the sequence.
        This should be specified for train and dev examples, but not for test
        examples.
    """

    guid: str
    words: List[str]
    labels: Optional[List[str]]


@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"
    tune = "tune"


class TokenClassificationDataSet(Dataset):
    """TokenClassificationDataSet PyTorch Dataset
    Args:
        sentences (:obj:`list` of :obj:`list` of :obj:`str`): The input
            sentences.
        tokenizer (:obj:`PreTrainedTokenizer`): Bert's pretrained tokenizer.
        labels (:obj:`list` of :obj:`str`): The labels which the model was
            trained to classify.
        max_seq_length (:obj:`int`):  Maximum sentence length.
    """

    def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            labels: List[str],
            model_type: str,
            max_seq_length: Optional[int] = None,
            mode: Split = Split.train
        ):
            examples = read_examples_from_file(data_dir, mode)
            # Use cross entropy ignore_index as padding label id so that only
            # real label ids contribute to the loss later.
            self.pad_token_label_id = nn.CrossEntropyLoss().ignore_index
            self.features = self._featurize_input(
                examples,
                labels,
                max_seq_length,
                tokenizer,
                cls_token=tokenizer.cls_token,
                sep_token=tokenizer.sep_token,
                pad_token=tokenizer.pad_token_id,
                pad_token_segment_id=tokenizer.pad_token_type_id,
                pad_token_label_id=self.pad_token_label_id,
            )

    def _featurize_input(self, examples, label_list, max_seq_length,
                        tokenizer, cls_token="[CLS]", cls_token_segment_id=0,
                        sep_token="[SEP]", pad_token=0, pad_token_segment_id=0,
                        pad_token_label_id=-100, sequence_a_segment_id=0,
                        mask_padding_with_zero=True):
        """Featurizes the input which will be fed to the fine-tuned BERT model.
        Args:
            examples (:obj:`list` of :obj:`InputExample`): list of
                InputExample objects.
            label_list (:obj:`list` of :obj:`str`): The labels which the model
                was trained to classify.
            max_seq_length (:obj:`int`):  Maximum sequence length.
            tokenizer (:obj:`PreTrainedTokenizer`): Bert's pretrained
                tokenizer.
            cls_token (:obj:`str`): BERT's CLS token. Defaults to [CLS].
            cls_token_segment_id (:obj:`int`): BERT's CLS token segment id.
                Defaults to 0.
            sep_token (:obj:`str`): BERT's CLS token. Defaults to [SEP].
            pad_token (:obj:`int`): BERT's pading token. Defaults to 0.
            pad_token_segment_id (:obj:`int`): BERT's pading token segment id.
                Defaults to 0.
            pad_token_label_id (:obj:`int`): BERT's pading token label id.
                Defaults to -100.
            sequence_a_segment_id (:obj:`int`): BERT's segment id.
                Defaults to 0.
            mask_padding_with_zero (:obj:`bool`): Whether to masks the padding
                tokens with zero or not. Defaults to True.
        Returns:
            obj:`list` of :obj:`Dict`: list of dicts of the needed features.
        """

        label_map = {label: i for i, label in enumerate(label_list)}
        features = []

        for sent_id, sentence in enumerate(examples):
            tokens = []
            label_ids = []

            for word, label in zip(sentence.words, sentence.labels):
                word_tokens = tokenizer.tokenize(word)
                # bert-base-multilingual-cased sometimes output "nothing ([])
                # when calling tokenize with just a space.
                if len(word_tokens) > 0:
                    tokens.append(word_tokens)
                    # Use the real label id for the first token of the word,
                    # and padding ids for the remaining tokens
                    label_ids.append([label_map[label]] +
                                     [pad_token_label_id] *
                                     (len(word_tokens) - 1))

            token_segments = []
            token_segment = []
            label_ids_segments = []
            label_ids_segment = []
            num_word_pieces = 0
            seg_seq_length = max_seq_length - 2

            # Dealing with empty sentences
            if len(tokens) == 0:
                data = self._add_special_tokens(token_segment,
                                                label_ids_segment,
                                                tokenizer,
                                                max_seq_length,
                                                cls_token,
                                                sep_token, pad_token,
                                                cls_token_segment_id,
                                                pad_token_segment_id,
                                                pad_token_label_id,
                                                sequence_a_segment_id,
                                                mask_padding_with_zero)
                # Adding sentence id
                data['sent_id'] = sent_id
                features.append(data)
            else:
                # Chunking the tokenized sentence into multiple segments
                # if it's longer than max_seq_length - 2
                for idx, word_pieces in enumerate(tokens):
                    if num_word_pieces + len(word_pieces) > seg_seq_length:
                        data = self._add_special_tokens(token_segment,
                                                        label_ids_segment,
                                                        tokenizer,
                                                        max_seq_length,
                                                        cls_token,
                                                        sep_token, pad_token,
                                                        cls_token_segment_id,
                                                        pad_token_segment_id,
                                                        pad_token_label_id,
                                                        sequence_a_segment_id,
                                                        mask_padding_with_zero)
                        # Adding sentence id
                        data['sent_id'] = sent_id
                        features.append(data)

                        token_segments.append(token_segment)
                        label_ids_segments.append(label_ids_segment)
                        token_segment = list(word_pieces)
                        label_ids_segment = list(label_ids[idx])
                        num_word_pieces = len(word_pieces)
                    else:
                        token_segment.extend(word_pieces)
                        label_ids_segment.extend(label_ids[idx])
                        num_word_pieces += len(word_pieces)

                # Adding the last segment
                if len(token_segment) > 0:
                    data = self._add_special_tokens(token_segment,
                                                    label_ids_segment,
                                                    tokenizer,
                                                    max_seq_length,
                                                    cls_token,
                                                    sep_token, pad_token,
                                                    cls_token_segment_id,
                                                    pad_token_segment_id,
                                                    pad_token_label_id,
                                                    sequence_a_segment_id,
                                                    mask_padding_with_zero)
                    # Adding sentence id
                    data['sent_id'] = sent_id
                    features.append(data)

                    token_segments.append(token_segment)
                    label_ids_segments.append(label_ids_segment)

                # DEBUG: Making sure we got all segments correctly
                # assert sum([len(_) for _ in label_ids_segments]) == \
                #        sum([len(_) for _ in label_ids])

                # assert sum([len(_) for _ in token_segments]) == \
                #        sum([len(_) for _ in tokens])

        return features

    def _add_special_tokens(self, tokens, label_ids, tokenizer, max_seq_length,
                            cls_token, sep_token, pad_token,
                            cls_token_segment_id, pad_token_segment_id,
                            pad_token_label_id, sequence_a_segment_id,
                            mask_padding_with_zero):

        _tokens = list(tokens)
        _label_ids = list(label_ids)

        _tokens += [sep_token]
        _label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(_tokens)

        _tokens = [cls_token] + _tokens
        _label_ids = [pad_token_label_id] + _label_ids
        segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(_tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only
        # real tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        input_ids += [pad_token] * padding_length
        input_mask += [0 if mask_padding_with_zero else 1] * padding_length
        segment_ids += [pad_token_segment_id] * padding_length
        _label_ids += [pad_token_label_id] * padding_length

        return {'input_ids': torch.tensor(input_ids),
                'attention_mask': torch.tensor(input_mask),
                'token_type_ids': torch.tensor(segment_ids),
                'label_ids': torch.tensor(_label_ids)}

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]


def read_examples_from_file(data_dir, mode: Union[Split, str]) -> List[InputExample]:
    if isinstance(mode, Split):
        mode = mode.value

    file_path = os.path.join(data_dir, f"{mode}.txt")
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples.append(InputExample(guid=f"{mode}-{guid_index}",
                                    words=words, labels=labels))
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
                    labels.append("C") # place holder
        if words:
            examples.append(InputExample(guid=f"{mode}-{guid_index}",
                            words=words, labels=labels))
    return examples


def get_labels(path: str):
    with open(path, "r") as f:
        labels = f.read().splitlines()
    return labels