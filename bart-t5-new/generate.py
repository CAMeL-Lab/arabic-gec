#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import disable_caching, Dataset
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW


import evaluate
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2SeqGEC,
    HfArgumentParser,
    M2M100Tokenizer,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from utils.postprocess import remove_pnx
from utils.m2scorer import m2scorer
from aligner.aligner import align
import re
import json


logger = logging.getLogger(__name__)

# A list of all multilingual tokenizer which require src_lang and tgt_lang attributes.
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast, M2M100Tokenizer]

torch.cuda.empty_cache()
datasets.disable_caching()


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    source_lang: str = field(default=None, metadata={"help": "Source language id for translation."})
    target_lang: str = field(default=None, metadata={"help": "Target language id for translation."})
    use_ged_tags: bool = field(default=None, metadata={"help": "To use ged tags or not."})
    preprocess_merges: bool = field(default=None, metadata={"help": "To preprocess Merges before inference."})
    m2_edits: str = field(default=None, metadata={"help": "Path to gold m2 edits for evaluation."})
    m2_edits_nopnx: str = field(default=None, metadata={"help": "Path to gold m2 edits without pnx for evaluation."})


    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the metrics (sacrebleu) on a jsonlines file."},
    )
    prediction_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional file to write the predictions to."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    num_return_sequences: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "Number of generated sequences. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the :obj:`decoder_start_token_id`.Useful for"
                " multilingual models like :doc:`mBART <../model_doc/mbart>` where the first generated token needs to"
                " be the target language token.(Usually it is the target language token)"
            )
        },
    )

    def __post_init__(self):
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_gec", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)


    # Set seed before initializing model.
    set_seed(training_args.seed)

    logger.info(f'Loading dataset from {data_args.test_file}')

    with open(data_args.test_file) as f:
        raw_data = [json.loads(l) for l in f.readlines()]

    dataset_dict = {'raw': [ex['raw'] for ex in raw_data],
                    'cor':  [ex['cor'] for ex in raw_data],
                    'ged_tags':  [ex['ged_tags'] for ex in raw_data]
                    }
    # for ex in raw_data:
    #     if 'ged_tags' in ex:
    #         dataset_dict.append(
    #                             {
    #                             'raw': ex['raw'],
    #                             'cor':  ex['gec']['cor'],
    #                             'ged_tags':  ex['gec']['ged_tags']
    #                             }
    #                         )
    #     else:
    #         dataset_dict.append(
    #                             {
    #                             'raw': ex['gec']['raw'],
    #                             'cor':  ex['gec']['cor']
    #                             }
    #                         )


    predict_dataset = Dataset.from_dict(dataset_dict)
    column_names = predict_dataset.column_names

    raw_predict_dataset = Dataset.from_dict(dataset_dict)

    logger.info(f'Loading model from {model_args.model_name_or_path}')

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=None
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=None,
        use_fast=model_args.use_fast_tokenizer
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=None
    )

    model.resize_token_embeddings(len(tokenizer))

    # preparing ged tags if the are provided
    if data_args.use_ged_tags:
        ged_label2id_map = config.ged_label2id
        ged_id2label_map  = config.ged_id2label
        num_labels = config.ged_num_labels


    # Set decoder_start_token_id
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.target_lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.target_lang)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""


    # For translation we set the codes of our source and target languages (only useful for mBART, the others will
    # ignore those attributes).
    if isinstance(tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
        assert data_args.target_lang is not None and data_args.source_lang is not None, (
            f"{tokenizer.__class__.__name__} is a multilingual tokenizer which requires --source_lang and "
            "--target_lang arguments."
        )

        tokenizer.src_lang = data_args.source_lang
        tokenizer.tgt_lang = data_args.target_lang

        # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
        # as the first generated token. We ask the user to explicitly provide this as --forced_bos_token argument.
        forced_bos_token_id = (
            tokenizer.lang_code_to_id[data_args.forced_bos_token] if data_args.forced_bos_token is not None else None
        )
        model.config.forced_bos_token_id = forced_bos_token_id

    # Get the language codes for input/target.
    source_lang = data_args.source_lang.split("_")[0]
    target_lang = data_args.target_lang.split("_")[0]

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False


    def preprocess_function(examples):

        inputs = examples['raw']
        ged_tags = examples['ged_tags'] if 'ged_tags' in examples else None


        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)


        # Converting the ged labels to ids if tags were provided
        if data_args.use_ged_tags:

            features = featurize_ged(tokenizer, inputs, ged_tags, ged_label2id_map,
                                     is_t5='t5' in model_args.model_name_or_path.lower(),
                                     do_preprocess=data_args.preprocess_merges)


            if data_args.preprocess_merges:
                input_ids = [features[i]['input_ids'] for i in range(len(features))]
                attention_mask = [features[i]['attention_mask'] for i in range(len(features))]

                model_inputs["input_ids"] = input_ids
                model_inputs["attention_mask"] = attention_mask

            else:

                # sanity checking if the featurization was done correctly
                gold_ids = [model_inputs['input_ids'][i] for i in range(len(model_inputs['input_ids']))]
                gold_ids_check = [features[i]['input_ids'] for i in range(len(features))]

                attention_mask = [model_inputs['attention_mask'][i] for i in range(len(model_inputs['attention_mask']))]
                attention_mask_check = [features[i]['attention_mask'] for i in range(len(features))]

                assert gold_ids_check == gold_ids
                assert attention_mask == attention_mask_check


            model_inputs["ged_tags"] = [features[i]['ged_labels_ids']
                                          for i in range(len(features))]


        return model_inputs


    max_target_length = data_args.val_max_target_length


    with training_args.main_process_first(desc="prediction dataset map pre-processing"):
        predict_dataset = predict_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=column_names,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=False,
            desc="Running tokenizer on prediction dataset",
        )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2SeqGEC(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )


    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )

    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    num_return_sequences = data_args.num_return_sequences

    gen_kwargs = {'num_beams': num_beams, 'max_length': max_length,
                  'num_return_sequences': num_return_sequences,
                  'no_repeat_ngram_size': 0, 'early_stopping': False
                  }

    logger.info("*** Predict ***")
    logger.info(f"Running prediction on {data_args.test_file}")

    # dataloader = trainer.get_test_dataloader(predict_dataset)
    dataloader = DataLoader(predict_dataset,
                            batch_size=training_args.per_device_eval_batch_size,
                            collate_fn=data_collator)

    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.eval()


    generated_dataset = []


    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if v is not None}

            if "attention_mask" in batch:
                gen_kwargs["attention_mask"] = batch.get("attention_mask", None)

            # providing ged tags to generation utils
            if "ged_tags" in batch:
                gen_kwargs["ged_tags"] = batch["ged_tags"]

            predictions = model.generate(
                batch['input_ids'],
                **gen_kwargs,
            )

            generated_tokens = tokenizer.batch_decode(
                    predictions, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            generated_dataset.extend(generated_tokens)


    if num_return_sequences == 1:
        generated_dataset = [pred.strip() for pred in generated_dataset]

        output_prediction_file = os.path.join(training_args.output_dir,
                                                data_args.prediction_file+f'.txt')


        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            writer.write("\n".join(generated_dataset))
            writer.write("\n")

        # removing the pnx from the generated outputs
        generated_dataset_nopnx = remove_pnx(generated_dataset)

        with open(output_prediction_file+'.nopnx', "w", encoding="utf-8") as writer:
            writer.write("\n".join(generated_dataset_nopnx))
            writer.write("\n")

        # running the m2 evaluation
        logger.info("*** Running M2 Evaluation ***")
        m2scorer.evaluate(output_prediction_file, data_args.m2_edits, timeout=30)

        # running the m2 evaluation without pnx 
        logger.info("*** Running M2 Evaluation (No Pnx) ***")
        m2scorer.evaluate(output_prediction_file+'.nopnx', data_args.m2_edits_nopnx, timeout=30)


        # running the m2 evaluation without pnx

        # # last steps for post_processing: pnx tokenization and m2 optim
        # post_processed_sents = postprocess(src_sents=[ex['raw'] for ex in raw_predict_dataset],
        #                                     preds_sents=generated_dataset)

        # with open(output_prediction_file+'.pp', "w", encoding="utf-8") as writer:
        #     writer.write("\n".join(post_processed_sents))
        #     writer.write("\n")

        # # running the m2 evaluation
        # logger.info("*** Running M2 Evaluation ***")

        # # running eval on pp files
        # m2score = m2scorer.evaluate(output_prediction_file+'.pp', data_args.m2_edits)

        # with open(output_prediction_file+'.pp.eval.check', "w", encoding="utf-8") as writer:
        #     writer.write(f"Precision   : {m2score['Precision']}\n")
        #     writer.write(f"Recall      : {m2score['Recall']}\n")
        #     writer.write(f"F_1.0       : {m2score['F1']}\n")
        #     writer.write(f"F_0.5       : {m2score['F0.5']}\n")


        # # running eval on originally generated files with timeout
        # m2score_timeout = m2scorer.evaluate(output_prediction_file, data_args.m2_edits,
        #                                     timeout=30)

        # # skipping sents if needed
        # m2_skipped_sents = m2score_timeout['Skipped']

        # m2_pp_sents = []
        # for i in range(len(generated_dataset)):
        #     if i in m2_skipped_sents:
        #         m2_pp_sents.append(raw_predict_dataset[i]['raw'])
        #     else:
        #         m2_pp_sents.append(generated_dataset[i])

        # with open(output_prediction_file+'.m2.pp', "w", encoding="utf-8") as writer:
        #     writer.write("\n".join(m2_pp_sents))
        #     writer.write("\n")

        # with open(output_prediction_file+'.m2.pp.eval.check', "w", encoding="utf-8") as writer:
        #     writer.write(f"Precision   : {m2score_timeout['Precision']}\n")
        #     writer.write(f"Recall      : {m2score_timeout['Recall']}\n")
        #     writer.write(f"F_1.0       : {m2score_timeout['F1']}\n")
        #     writer.write(f"F_0.5       : {m2score_timeout['F0.5']}\n")


    else:
        generated_nbest = [[] for _ in range(num_return_sequences)]

        for seq in range(num_return_sequences):
            gen_n = [generated_dataset[seq + i] for i in range(0, len(generated_dataset), num_return_sequences)]
            generated_nbest[seq] = gen_n

        for i, gen_n in enumerate(generated_nbest):
            output_prediction_file = os.path.join(training_args.output_dir,
                                                  data_args.prediction_file+f'.{i+1}.txt')

            with open(output_prediction_file, "w", encoding="utf-8") as writer:
                writer.write("\n".join(gen_n))
                writer.write("\n")

            # last steps for post_processing: pnx tokenization and m2 optim
            post_processed_sents = postprocess(src_sents=[ex['raw'] for ex in raw_predict_dataset],
                                            preds_sents=gen_n)


            with open(output_prediction_file+'.pp', "w", encoding="utf-8") as writer:
                writer.write("\n".join(post_processed_sents))
                writer.write("\n")



def preprocess(words, labels):
    """
    Process words by solving merge errors
    """

    new_words = []
    new_labels = []

    i = 0
    while i < len(words):
        word = words[i]
        label = labels[i]

        # TODO: Sometimes we might see a single Merge-B label
        # that is not followed by Merge-I. We should handle those 
        # cases better

        if label == 'MERGE-B':
            new_word = []
            new_word.append(word)
            i += 1

            while  i < len(labels) and 'MERGE-I' in labels[i]:
                new_word.append(words[i])
                i += 1

            new_word = ''.join(new_word)
            new_words.append(new_word)
            new_labels.append('UC')

        elif label == 'DELETE':
            i += 1
            continue

        else:
            new_words.append(word)
            new_labels.append(label)
            i += 1

    assert len(new_words) == len(new_labels)

    return new_words, new_labels


def featurize_ged(tokenizer, inputs, labels, label_map, is_t5=False, do_preprocess=False):
    """
    Featurizes ged labels. Each subword that belongs to the same word
    gets the same ged label (i.e., each input sentence will have
    the same number of subwords and same number of ged labels)
    """

    features = []

    for i, (seq, seq_labels) in enumerate(zip(inputs, labels)):
        labels = []
        tokens = []

        words, ged_labels = seq.split(), seq_labels.split()
        assert len(words) == len(ged_labels)

        if do_preprocess:
            words, ged_labels = preprocess(words, ged_labels)

        for word, label in zip(words, ged_labels):
            word_tokens = tokenizer.tokenize(word)

            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                labels.extend([label for _ in range(len(word_tokens))])

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # converting the labels to label ids
        # for labels that we have not seen in training, they get pad_id
        label_ids = [label_map.get(label, label_map['<pad>']) for label in labels]

        # Adding bos and eos tokens to the input ids
        if is_t5:
            input_ids = input_ids + [tokenizer.eos_token_id]
            label_ids = label_ids + [label_map['UC']]
        else:
            input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
            label_ids = [label_map['UC']] + label_ids + [label_map['UC']]


        assert len(label_ids) == len(input_ids)

        attention_mask = [1 for _ in range(len(input_ids))]

        features.append({'input_ids': input_ids,
                         'ged_labels_ids': label_ids,
                         'attention_mask': attention_mask}
                        )

    return features


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
