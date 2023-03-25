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
    vectorized_labels = []

    label_ids = [0 for _ in range(len(label_map))]
    for sublabel in label.split('+'):
        label_ids[label_map[sublabel]] = 1


    return np.asarray(label_ids)


def eval(gold_labels, preds_labels, labels, mode=None, include_uc=True):
    uniq_combos = set([x for sublist in gold_labels for x in sublist])
    uniq_single = set([x for label in uniq_combos for x in label.split('+')])
    # labels = set(labels).union(uniq_single) - set(['UNK'])
    if include_uc:
        labels = set(labels).union(uniq_single)
        labels_map = {label: i for i, label in enumerate(sorted(labels))}

        flatten_gold = [label for sublist in gold_labels for label in sublist]
        flatten_pred = [label for sublist in preds_labels for label in sublist]

        flatten_gold_ = flatten_gold
        flatten_pred_ = flatten_pred
    else:
        labels = set(labels).union(uniq_single)
        labels_map = {label: i for i, label in enumerate(sorted(labels))}

        flatten_gold = [label for sublist in gold_labels for label in sublist]
        flatten_pred = [label for sublist in preds_labels for label in sublist]

        flatten_gold_ = [label for i, label in enumerate(flatten_gold) if flatten_gold[i] != 'UC']
        flatten_pred_ = [label for i, label in enumerate(flatten_pred) if flatten_gold[i] != 'UC']


    # taking out unk
    # flatten_gold_ = [label for i, label in enumerate(flatten_gold) if flatten_gold[i] != 'UNK']
    # flatten_pred_ = [label for i, label in enumerate(flatten_pred) if flatten_gold[i] != 'UNK']

    if mode == 'verbose':
        flatten_gold_simple = [label for i, label in enumerate(flatten_gold_) if '+' not in flatten_gold_[i]]
        flatten_pred_simple = [label for i, label in enumerate(flatten_pred_) if '+' not in flatten_gold_[i]]

        simple_metrics = eval_simple(flatten_gold_simple, flatten_pred_simple)

        flatten_gold_comb = [label for i, label in enumerate(flatten_gold_) if '+' in flatten_gold_[i]]
        flatten_pred_comb = [label for i, label in enumerate(flatten_pred_) if '+' in flatten_gold_[i]]

        vectorized_gold = [vectorize_labels(labels_map, label) for label in flatten_gold_comb]
        vectorized_pred = [vectorize_labels(labels_map, label) for label in flatten_pred_comb]

        complex_metrics = compute_metrics(vectorized_gold, vectorized_pred, labels)

        return {'combinations': complex_metrics,
                'single': simple_metrics}


    else:
        vectorized_gold = [vectorize_labels(labels_map, label) for label in flatten_gold_]
        vectorized_pred = [vectorize_labels(labels_map, label) for label in flatten_pred_]

        return compute_metrics(vectorized_gold, vectorized_pred, labels)


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


def compute_metrics(gold_list, preds_list, labels):
    # we will evaluate at the word-level
    # avg_f1, avg_f05, avg_precision, avg_recall = 0, 0, 0, 0

    # for i in range(len(gold_list)):
    #     gold = gold_list[i]
    #     pred = preds_list[i]

    #     # we dont care about evaluation when both the gold and the 
    #     # pred are zero
    #     _preds = []
    #     _gold = []

    #     for i in range(len(gold)):
    #         if pred[i] == 0 and gold[i] == 0:
    #             continue
    #         else:
    #             _preds.append(pred[i])
    #             _gold.append(gold[i])

    #     f1 = f1_score(_gold, _preds, zero_division=0)
    #     p =  precision_score(_gold, _preds, zero_division=0)
    #     r = recall_score(_gold, _preds, zero_division=0)

    #     avg_f1 += f1
    #     avg_precision += p
    #     avg_recall += r
    #     avg_f05 += (1 + 0.5**2) * p * r / (((0.5**2)*p) + r) if (p != 0 and r != 0) else 0


    # avg_f1 = avg_f1 / len(gold_list)
    # avg_precision = avg_precision / len(gold_list)
    # avg_recall = avg_recall / len(gold_list)
    # avg_f05 = avg_f05 / len(gold_list)



    target_names = sorted(list(labels))

    # getting the f0.5 for each class
    avg_samples_p, avg_samples_r, avg_samples_f05, support = precision_recall_fscore_support(gold_list,
                                                                                            preds_list,
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
        assert avg_samples_p[i] == report[target]['precision']
        assert avg_samples_r[i] == report[target]['recall']

        report[target]['f0.5-score'] = avg_samples_f05[i]

    # adding the averages for f0.5
    for avg in f05_avgs:
        report[f'{avg} avg']['f0.5-score'] = f05_avgs[avg]

    return {'f0.5': report['samples avg']['f0.5-score'],
            'f1': report['samples avg']['f1-score'],
            'precision': report['samples avg']['precision'],
            'recall':  report['samples avg']['recall'],
            'report': report
            }


def read_data(path, mode=None):
    tags = []
    all_tags = []
    with open(path) as f:
        if mode == 'pred':
            for line in f.readlines():
                line = line.strip()
                if line:
                    tags.append('+'.join(sorted(line.split('+'))))
                else:
                    all_tags.append(tags)
                    tags = []
            if tags:
                all_tags.append(tags)

        else:
            for line in f.readlines():
                line = line.strip().split('\t')
                if len(line) > 1:
                    tags.append('+'.join(sorted(line[-1].split('+'))))
                else:
                    all_tags.append(tags)
                    tags = []
            if tags:
                all_tags.append(tags)

    return all_tags


def get_labels(path):
    with open(path, "r") as f:
        labels = f.read().splitlines()

    return labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred')
    parser.add_argument('--gold')
    parser.add_argument('--labels')
    parser.add_argument('--output')
    parser.add_argument('--mode')

    args = parser.parse_args()

    labels = get_labels(args.labels)
    pred_tags = read_data(args.pred, mode='pred')
    gold_tags = read_data(args.gold)

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
        results = eval(gold_tags, pred_tags, labels, include_uc=True)
        with open(args.output, "w") as f:
            for metric in ['f0.5', 'f1', 'precision', 'recall']:
                f.write("%s = %s\n" % (metric, results[metric]))

        df = pd.DataFrame(results['report']).transpose()
        df.iloc[:,[0,1,2,4,3]]
        # df.to_csv(args.output, mode='a', sep="\t")
        # changing the order of the df
        df.iloc[:,[0,1,2,4,3]].to_csv(args.output, mode='a', sep="\t")
