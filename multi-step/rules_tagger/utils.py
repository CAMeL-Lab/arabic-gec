import logging
import os
from dataclasses import dataclass
import json
from filelock import FileLock
from enum import Enum
from typing import List, Optional, Union

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


logger = logging.getLogger(__name__)


@dataclass
class InputExample:
    """
    A single training/test example for token classification.

    Args:
        guid: Unique id for the example.
        subwords: list. The subwords of the sequence.
        labels: (Optional) list. The labels for each word of the sequence.
        This should be specified for train and dev examples, but not for test
        examples.
    """

    guid: str
    subwords: List[str]
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


class GECTokenClassificationDataset(Dataset):

    features: List[InputFeatures]
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index

    # Use cross entropy ignore_index as padding label id so that only
    # real label ids contribute to the loss later.

    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizer,
        labels: List[str],
        max_seq_length: Optional[int] = None,
        mode: Split = Split.train
    ):

        logger.info(f"Creating features from dataset file at {data_dir}")
        examples = read_examples_from_file(data_dir, mode)
        self.features = convert_examples_to_features(
                    examples,
                    labels,
                    max_seq_length,
                    tokenizer,
                    cls_token=tokenizer.cls_token,
                    cls_token_segment_id=0,
                    sep_token=tokenizer.sep_token,
                    pad_token=tokenizer.pad_token_id,
                    pad_token_segment_id=tokenizer.pad_token_type_id,
                    pad_token_label_id=self.pad_token_label_id
                )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]



def read_examples_from_file(data_dir, mode: Union[Split, str]) -> List[InputExample]:
    if isinstance(mode, Split):
        mode = mode.value

    file_path = os.path.join(data_dir, f"{mode}.txt")
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        subwords = []
        labels = []
        for line in f:
            if line == "" or line == "\n":
                if subwords:
                    examples.append(InputExample(guid=f"{mode}-{guid_index}",
                                    subwords=subwords, labels=labels))
                    guid_index += 1
                    subwords = []
                    labels = []
            else:
                splits = line.split("\t")
                subwords.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    # This is needed to get around the Trainer evaluation
                    labels.append('["Rule", {"edits": [["UC"]]}, {}]') # place holder

        if subwords:
            examples.append(InputExample(guid=f"{mode}-{guid_index}",
                            subwords=subwords, labels=labels))

    return examples


def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_seq_length: int,
    tokenizer: PreTrainedTokenizer,
    cls_token="[CLS]",
    cls_token_segment_id=0,
    sep_token="[SEP]",
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True
) -> List[InputFeatures]:
    """ Loads a data file into a list of `InputFeatures'"""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10_000 == 0:
            logger.info("Processing example %d of %d", ex_index, len(examples))

        tokens = example.subwords
        label_ids = []
        for label in example.labels:
            if label == '["Rule", {"edits": [["UNK"]]}, {}]':
                label_ids.append(-200)
            else:
                label_ids.append(label_map[label])

        assert len(label_ids) == len(tokens)


        special_tokens_count = tokenizer.num_special_tokens_to_add()
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]


        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        tokens = [cls_token] + tokens
        label_ids = [pad_token_label_id] + label_ids
        segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        input_ids += [pad_token] * padding_length
        input_mask += [0 if mask_padding_with_zero else 1] * padding_length
        segment_ids += [pad_token_segment_id] * padding_length
        label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        if "token_type_ids" not in tokenizer.model_input_names:
            segment_ids = None

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=input_mask,
                          token_type_ids=segment_ids,
                          label_ids=label_ids))

    return features


def get_labels(path: str) -> List[str]:
    with open(path, "r") as f:
        labels = f.read().splitlines()

    return labels



if __name__ == '__main__':
    labels = get_labels('/scratch/ba63/gec/data/rules-tagger-data/rules.txt')
    from transformers import AutoTokenizer
    import pdb; pdb.set_trace()
    tokenizer = AutoTokenizer.from_pretrained('/scratch/ba63/BERT_models/bert-base-arabic-camelbert-msa')
    dataset = GECTokenClassificationDataset(
                data_dir='/scratch/ba63/gec/data/rules-tagger-data',
                tokenizer=tokenizer,
                labels=labels,
                max_seq_length=256,
                mode=Split.train
            )

    pdb.set_trace()