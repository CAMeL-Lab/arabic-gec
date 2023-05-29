from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    precision_recall_fscore_support
    )
import argparse
import os
import numpy as np
import pandas as pd


def vectorize_labels(label_map, label):
    """Represents a label as a one-hot vector

    Args:
        label_map (dict): label map dictionary
        label (str): label

    Returns:
        a one-hot vector representation for the label
    """
    vectorized_labels = []

    label_ids = [0 for _ in range(len(label_map))]
    for sublabel in label.split('+'):
        label_ids[label_map[sublabel]] = 1

    return np.asarray(label_ids)


def eval(gold_labels, preds_labels, labels, mode=None):
    """Main evaluation function

    Args:
        gold_labels (list of list of str): gold labels
        preds_labels (list of list of str): prediction labels
        labels (list): list of single labels in the output space
        mode (str, optional): a flag to output detailed evaluation.
                              on combination errors or not. Defaults to None.

    Returns:
        evaluation metrics as a dict.
    """
    uniq_labels = set([x for sublist in gold_labels for x in sublist])
    # We will do the evaluation on single-label space
    uniq_single = set([x for label in uniq_labels for x in label.split('+')])

    labels = set(labels).union(uniq_single)
    labels_map = {label: i for i, label in enumerate(sorted(labels))}

    flatten_gold = [label for sublist in gold_labels for label in sublist]
    flatten_pred = [label for sublist in preds_labels for label in sublist]

    if mode == 'verbose':
        flatten_gold_simple = [label for i, label in enumerate(flatten_gold_)
                               if '+' not in flatten_gold_[i]]
        flatten_pred_simple = [label for i, label in enumerate(flatten_pred_)
                               if '+' not in flatten_gold_[i]]

        simple_metrics = eval_simple(flatten_gold_simple, flatten_pred_simple)

        flatten_gold_comb = [label for i, label in enumerate(flatten_gold_)
                              if '+' in flatten_gold_[i]]
        flatten_pred_comb = [label for i, label in enumerate(flatten_pred_)
                              if '+' in flatten_gold_[i]]

        vectorized_gold = [vectorize_labels(labels_map, label) for label in flatten_gold_comb]
        vectorized_pred = [vectorize_labels(labels_map, label) for label in flatten_pred_comb]

        complex_metrics = compute_metrics(vectorized_gold, vectorized_pred, labels)

        return {'combinations': complex_metrics,
                'single': simple_metrics}

    else:
        vectorized_gold = [vectorize_labels(labels_map, label) for label in flatten_gold]
        vectorized_pred = [vectorize_labels(labels_map, label) for label in flatten_pred]
        full_eval = compute_metrics(vectorized_gold, vectorized_pred, labels)
        non_uc_bin_eval = binarized_eval(flatten_gold, flatten_pred, beta=0.5)

        return full_eval, non_uc_bin_eval


def eval_simple(flatten_gold, flatten_pred):
    accuracy = accuracy_score(flatten_gold, flatten_pred)

    labels = sorted(list(set(flatten_gold)))

    acc_per_tag = {}
    for label in labels:
        gold_tags = [x for i, x in enumerate(flatten_gold) if x == label]
        pred_tags = [x for i, x in enumerate(flatten_pred) if flatten_gold[i] == label]
        acc_per_tag[label] = {'accuracy': accuracy_score(gold_tags, pred_tags),
                              'support': len(gold_tags)}

    return {'f1': accuracy,
            'precision': accuracy,
            'recall': accuracy,
            'report': acc_per_tag
            }

def binarized_eval(gold_labels, pred_labels, beta=0.5):
    """Compute evaluation metrics for non-UC classes in a binarized space

    Args:
        gold_labels (list of str): list of gold labels
        pred_labels (list of str): list of pred labels
        beta (float, optional): F score beta. Defaults to 0.5.

    Returns:
        precision, recall and F-beta scores
    """
    assert len(gold_labels) == len(pred_labels)
    # Turning labels into UC and I
    bin_gold = ['I' if label != 'UC' else label for label in gold_labels]
    bin_pred = ['I' if label != 'UC' else label for label in pred_labels]

    assert len(bin_gold) == len(bin_pred)

    lname = 'I'
    tp, fp, fn = 0, 0, 0

    for i in range(len(bin_gold)):
        # True Positive
        if bin_pred[i] == bin_gold[i] == lname: tp += 1
        # Non-matching labels
        if bin_pred[i] != bin_gold[i]:
            # False positive: pred says there's an error but gold is uc
            if bin_pred[i] == lname: fp += 1
            # False negative: pred is uc but gold says there's an error
            if bin_gold[i] == lname: fn += 1

    p = float(tp)/(tp+fp) if fp else 1.0
    r = float(tp)/(tp+fn) if fn else 1.0
    f = float((1+(beta**2))*p*r)/(((beta**2)*p)+r) if p+r else 0.0

    return {'precision': p,
            'recall': r,
            'f0.5': f
            }

def compute_metrics(gold_list, preds_list, labels):
    """Computes the evaluation metrics

    Args:
        gold_list (list of list of int): one-hot encoded gold labels
        preds_list (list of list of int): one-hot encoded pred labels
        labels (list of str): list of single labels in the output space

    Returns:
        evaluation metrics as a dict
    """
    # we will evaluate at the word-level
    target_names = sorted(list(labels))

    # getting the p, r, f0.5 for each class
    avg_ex_p, avg_ex_r, avg_ex_f05, support = precision_recall_fscore_support(gold_list,
                                                                              preds_list,
                                                                              average=None,
                                                                              zero_division=0,
                                                                              beta=0.5)

    # getting the average f0.5 in all flavors
    f05_avgs = {'micro': 0, 'macro': 0, 'weighted': 0, 'samples': 0}
    for avg in f05_avgs:
        _, _, avg_f05, _ = precision_recall_fscore_support(gold_list, preds_list,
                                                            average=avg,
                                                            beta=0.5,
                                                            zero_division=0)
        f05_avgs[avg] = avg_f05

    report = classification_report(gold_list, preds_list,
                                   target_names=target_names,
                                   zero_division=0, output_dict=True)

    # adding f0.5 for each class
    for i, target in enumerate(target_names):
        assert support[i] == report[target]['support']
        assert avg_ex_p[i] == report[target]['precision']
        assert avg_ex_r[i] == report[target]['recall']

        report[target]['f0.5-score'] = avg_ex_f05[i]

    # adding the averages for f0.5
    for avg in f05_avgs:
        report[f'{avg} avg']['f0.5-score'] = f05_avgs[avg]

    return {'f0.5': report['samples avg']['f0.5-score'],
            'f1': report['samples avg']['f1-score'],
            'precision': report['samples avg']['precision'],
            'recall':  report['samples avg']['recall'],
            'report': report
            }

def read_data(path):
    """Reads gold and predicted labels

    Args:
        path (str): path of a tsv file containing gold and predicted labels

    Returns:
        tuple of gold and predicted labels respectively.
    """
    pred_tags = []
    gold_tags = []
    all_gold_tags = []
    all_pred_tags = []

    with open(path) as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            if len(line) > 1:
                gold = '+'.join(sorted(line[1].split('+')))
                pred = '+'.join(sorted(line[2].split('+')))
                gold_tags.append(gold)
                pred_tags.append(pred)
            else:
                all_gold_tags.append(gold_tags)
                all_pred_tags.append(pred_tags)
                gold_tags = []
                pred_tags = []

        if pred_tags and gold_tags:
            all_gold_tags.append(gold_tags)
            all_pred_tags.append(pred_tags)

    return all_gold_tags, all_pred_tags


def get_labels(path):
    with open(path, "r") as f:
        labels = f.read().splitlines()

    return labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    parser.add_argument('--labels')
    parser.add_argument('--output')
    parser.add_argument('--mode')

    args = parser.parse_args()

    labels = get_labels(args.labels)
    gold_tags, pred_tags = read_data(args.data)

    if args.mode == 'verbose':
        results = eval(gold_tags, pred_tags, labels, mode=args.mode)

        with open(args.output, "w") as f:
            f.write('Tag Combinations Metrics:\n\n')
            for metric in ['f1', 'precision', 'recall']:
                f.write("%s = %s\n" % (metric, results['combinations'][metric]))

            f.write("\n\n")
            df = pd.DataFrame(results['combinations']['report']).transpose()
            df.to_csv(f, sep="\t")

            f.write('\n\n')
            f.write('Single Tag Metrics:\n\n')

        with open(args.output, "a") as f:
            for metric in ['f1', 'precision', 'recall']:
                f.write("%s = %s\n" % (metric, results['single'][metric]))

            f.write("\n\n")
            df = pd.DataFrame(results['single']['report']).transpose()
            df.to_csv(f, sep="\t")

    else:
        full_eval, non_uc_bin_eval = eval(gold_tags, pred_tags, labels)
        with open(args.output+'.txt', "w") as f:
            for metric in ['f0.5', 'f1', 'precision', 'recall']:
                f.write("%s = %s\n" % (metric, full_eval[metric]))

        with open(args.output+'.bin.txt', "w") as f:
            for metric in ['f0.5', 'precision', 'recall']:
                f.write("%s = %s\n" % (metric, non_uc_bin_eval[metric]))

        df = pd.DataFrame(full_eval['report']).transpose()
        df.iloc[:,[0,1,2,4,3]]

        # changing the order of the df
        df.iloc[:,[0,1,2,4,3]].to_csv(args.output+'.txt', mode='a', sep="\t")
