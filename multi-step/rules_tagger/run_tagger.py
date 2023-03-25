import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)

from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from utils import GECTokenClassificationDataset, Split, get_labels
from model import BertForTokenClassification


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are
    going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from "
                          "huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if "
                                        "not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if "
                                        "not the same as model_name"}
    )
    use_fast: bool = field(default=False, metadata={"help": "Set this flag to "
                                                            "use fast "
                                                            "tokenization."})

    # If you want to tweak more attributes on your tokenizer, you should do it
    # in a distinct script, or just modify its tokenizer_config.json.

    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the "
                                        "pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for
    training and eval.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .txt files "
                          "for a CoNLL-2003-formatted task."}
    )
    labels: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a file containing all labels."},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after "
            "tokenization. Sequences longer than this will be truncated, "
            "sequences shorter will be padded."
        },
    )
    pred_mode: Optional[str] = field(
        default=None, metadata={"help": "Prediction mode to get the actual "
                                        "token predictions on dev or test."}
    )
    pred_output_file: Optional[str] = field(
        default=None, metadata={"help": "Predictions output file."}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments,
                               DataTrainingArguments,
                               TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a
        # json file, let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
                                                    json_file=os.path.abspath(
                                                                 sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists "
            "and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=(logging.INFO if training_args.local_rank in [-1, 0]
               else logging.WARN),
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, "
        "16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Prepare task
    labels = get_labels(data_args.labels)
    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)


    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can
    # concurrently download model & vocab.

    config = AutoConfig.from_pretrained(
        (model_args.config_name if model_args.config_name
            else model_args.model_name_or_path),
        num_labels=num_labels,
        id2label=label_map,
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        (model_args.tokenizer_name if model_args.tokenizer_name
            else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast,
    )

    # Get datasets
    train_dataset = (
        GECTokenClassificationDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            max_seq_length=data_args.max_seq_length,
            mode=Split.train
        )
        if training_args.do_train
        else None
    )

    eval_dataset = (
        GECTokenClassificationDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            max_seq_length=data_args.max_seq_length,
            mode=Split.dev
        )
        if training_args.do_eval
        else None
    )


    model = BertForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir
    )


    def align_predictions(predictions: np.ndarray,
                          label_ids: np.ndarray) -> Tuple[List[int], List[int]]:
        import torch
        preds = np.argmax(predictions, axis=2)

        batch_size, seq_len = preds.shape
        preds_list = [[] for _ in range(batch_size)]


        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    preds_list[i].append(label_map[preds[i][j]])

        return preds_list


    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=(model_args.model_name_or_path
                        if os.path.isdir(model_args.model_name_or_path)
                        else None)
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        result = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir,
                                        "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

            results.update(result)

    # Predict
    if training_args.do_predict:
        pred_mode = data_args.pred_mode

        if pred_mode == "test":
            pred_data = Split.test
        elif pred_mode == "dev":
            pred_data = Split.dev
        elif pred_mode == "train":
            pred_data = Split.train
        elif pred_mode == "tune":
            pred_data = Split.tune

        test_dataset = GECTokenClassificationDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            max_seq_length=data_args.max_seq_length,
            mode=pred_data
        )

        predictions, label_ids, _ = trainer.predict(test_dataset)

        preds_list = align_predictions(predictions, label_ids)

        # Save predictions
        if data_args.pred_output_file:
            output_test_predictions_file = os.path.join(training_args.output_dir,
                                                        data_args.pred_output_file)
        else:
            output_test_predictions_file = os.path.join(training_args.output_dir,
                                                        pred_mode+"_predictions.txt")

        if trainer.is_world_process_zero():
            with open(output_test_predictions_file, "w") as writer:
                for example in preds_list:
                    for label in example:
                        writer.write(label)
                        writer.write('\n')
                    writer.write('\n')


    return results


if __name__ == "__main__":
    main()
