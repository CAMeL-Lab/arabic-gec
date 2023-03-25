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
import torch
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed
)
from utils import TokenClassificationDataSet, Split, get_labels
from model import BertForTokenClassificationSingleLabel
from torch.utils.data import DataLoader


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

    add_class_weights: bool = field(
        default=False, metadata={"help": "Whether to weigh classes during "
                                        "training or not."}
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


def predict(model, args, dataset):
    data_loader = DataLoader(dataset,
                            batch_size=args.per_device_eval_batch_size,
                            shuffle=False)

    label_ids = None
    preds = None
    sent_ids = None

    model.to(args.device)
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}

            inputs = {'input_ids': batch['input_ids'],
                      'token_type_ids': batch['token_type_ids'],
                      'attention_mask': batch['attention_mask']}

            label_ids = (batch['label_ids'] if label_ids is None
                            else torch.cat((label_ids, batch['label_ids'])))

            sent_ids = (batch['sent_id'] if sent_ids is None
                        else torch.cat((sent_ids, batch['sent_id'])))

            logits = model(**inputs)[0]

            preds = logits if preds is None else torch.cat((preds, logits),
                                                            dim=0)

    return preds.cpu().numpy(), label_ids.cpu().numpy(), sent_ids.cpu().numpy()

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
        TokenClassificationDataSet(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            mode=Split.train
        )
        if training_args.do_train
        else None
    )

    eval_dataset = (
        TokenClassificationDataSet(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            mode=Split.dev
        )
        if training_args.do_eval
        else None
    )

    model = BertForTokenClassificationSingleLabel.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        class_weights=train_dataset.class_weights if model_args.add_class_weights else None
    )

    def align_predictions(predictions: np.ndarray,
                          label_ids: np.ndarray,
                          sent_ids: np.ndarray) -> Tuple[List[int], List[int]]:
        preds = np.argmax(predictions, axis=2)

        batch_size, seq_len = preds.shape
        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(label_map[label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        # Collating the predicted labels based on the sentence ids
        final_preds_list = [[] for _ in range(len(set(sent_ids)))]
        final_label_list =  [[] for _ in range(len(set(sent_ids)))]
        for i, id in enumerate(sent_ids):
            final_preds_list[id].extend(preds_list[i])
            final_label_list[id].extend(out_label_list[i])

        return final_preds_list, final_label_list


    def compute_metrics(preds_list, out_label_list) -> Dict:


        # Flatten the preds_list and out_label_list
        preds_list = [p for sublist in preds_list for p in sublist]
        out_label_list = [p for sublist in out_label_list for p in sublist]

        metrics = {
            "accuracy": accuracy_score(out_label_list, preds_list),
            # "precision_micro": precision_score(out_label_list, preds_list,
            #                                     average="micro"),
            # "recall_micro": recall_score(out_label_list, preds_list,
            #                                 average="micro"),
            # "f1_micro": f1_score(out_label_list, preds_list,
            #                         average="micro"),
            # "precision_macro": precision_score(out_label_list, preds_list,
            #                                     average="macro"),
            # "recall_macro": recall_score(out_label_list, preds_list,
            #                                 average="macro"),
            # "f1_macro": f1_score(out_label_list, preds_list,
            #                         average="macro"),
        }

        return metrics


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

        test_dataset = TokenClassificationDataSet(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            mode=pred_data
        )

        # predictions, label_ids, metrics = trainer.predict(test_dataset)
        predictions, label_ids, sent_ids = predict(model, training_args, test_dataset)

        preds_list, labels = align_predictions(predictions, label_ids, sent_ids)
        metrics = compute_metrics(preds_list, labels)

        output_test_results_file = os.path.join(training_args.output_dir,
                                                pred_mode+"_results.txt")
        if trainer.is_world_process_zero():
            with open(output_test_results_file, "w") as writer:
                for key, value in metrics.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

        # Save predictions
        output_test_predictions_file = os.path.join(training_args.output_dir,
                                                    pred_mode+"_predictions.txt")

        with open(output_test_predictions_file, "w") as f:
            for example in preds_list:
                for pred in example:
                    f.write(pred)
                    f.write('\n')
                f.write('\n')

    return results


if __name__ == "__main__":
    main()
