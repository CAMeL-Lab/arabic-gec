from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    f1_score,
    accuracy_score
    )
import argparse
import pandas as pd


def eval(gold_labels, preds_labels, labels):
    """Main evaluation function

    Args:
        gold_labels (list of list of str): gold labels
        preds_labels (list of list of str): prediction labels
        labels (list): list of output labels

    Returns:
        evaluation metrics as a dict.
    """

    flatten_gold = [label for sublist in gold_labels for label in sublist]
    flatten_pred = [label for sublist in preds_labels for label in sublist]

    metrics = compute_metrics(flatten_gold, flatten_pred, labels)
    binarized_metrics = binarized_eval(flatten_gold, flatten_pred)
    return metrics, binarized_metrics


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
    accuracy = accuracy_score(bin_gold, bin_pred)
    # p, r, f, err_support = precision_recall_fscore_support(bin_gold,
    #                                                         bin_pred,
    #                                                         labels=['I'],
    #                                                         average='macro',
    #                                                         zero_division=0,
    #                                                         beta=0.5)

    return {'precision': p,
            'recall': r,
            'f0.5': f,
            'accuracy': accuracy
            }

def compute_metrics(gold_list, preds_list, labels):
    """Computes the evaluation metrics

    Args:
        gold_list (list of list of int): one-hot encoded gold labels
        preds_list (list of list of int): one-hot encoded pred labels
        labels (list of str): list of the output labels

    Returns:
        evaluation metrics as a dict
    """
    err_labels = [x for x in labels if x != 'UC']
    # getting the macro p, r, f1, f0.5 for each class *without* UC
    err_p, err_r, err_f05, err_support = precision_recall_fscore_support(gold_list,
                                                                         preds_list,
                                                                         labels=err_labels,
                                                                         average='macro',
                                                                         zero_division=0,
                                                                         beta=0.5)
    err_f1 = f1_score(gold_list, preds_list, labels=err_labels, average='macro',
                      zero_division=0)


    # getting the macro p, r, f1, f0.5 for each class *inclduing* UC
    all_p, all_r, all_f05, all_support = precision_recall_fscore_support(gold_list,
                                                                         preds_list,
                                                                         labels=labels,
                                                                         average='macro',
                                                                         zero_division=0,
                                                                         beta=0.5)

    all_f1 = f1_score(gold_list, preds_list, labels=labels, average='macro',
                      zero_division=0)


    # getting possible p, r, f1, f0.5 for each class *including* UC
    p, r, f05, support = precision_recall_fscore_support(gold_list,
                                                         preds_list,
                                                         labels=labels,
                                                         average=None,
                                                         zero_division=0,
                                                         beta=0.5)

    # getting the average f0.5 in all flavors *including* UC
    f05_avgs = {'macro': 0, 'weighted': 0}
    for avg in f05_avgs:
        _, _, avg_f05, _ = precision_recall_fscore_support(gold_list,
                                                            preds_list,
                                                            average=avg,
                                                            labels=labels,
                                                            beta=0.5,
                                                            zero_division=0)
        f05_avgs[avg] = avg_f05


    report = classification_report(gold_list, preds_list,
                                   labels=labels,
                                   target_names=labels,
                                   zero_division=0,
                                   output_dict=True)

    # adding f0.5 for each class
    for i, target in enumerate(labels):
        assert support[i] == report[target]['support']
        assert p[i] == report[target]['precision']
        assert r[i] == report[target]['recall']

        report[target]['f0.5-score'] = f05[i]

    # adding the averages for f0.5
    for avg in f05_avgs:
        report[f'{avg} avg']['f0.5-score'] = f05_avgs[avg]


    return {'f0.5': all_f05,
            'f1': all_f1,
            'precision': all_p,
            'recall':  all_r,
            'err_f0.5': err_f05,
            'err_f1': err_f1,
            'err_precision': err_p,
            'err_recall':  err_r,
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

    args = parser.parse_args()
    labels = get_labels(args.labels)
    gold_tags, pred_tags = read_data(args.data)

    full_eval, non_uc_bin_eval = eval(gold_tags, pred_tags, labels)


    with open(args.output+'.bin.txt', "w") as f:
        for metric in ['precision', 'recall', 'f0.5', 'accuracy']:
            f.write(f'{metric} : {non_uc_bin_eval[metric]}\n')

    with open(args.output+'.txt', "w") as f:
        for metric in ['precision', 'recall', 'f1', 'f0.5',
                       'err_precision', 'err_recall', 'err_f1', 'err_f0.5']:
            f.write(f'{metric} : {full_eval[metric]}\n')

    df = pd.DataFrame(full_eval['report']).transpose()
    df.iloc[:,[0,1,2,4,3]]

    # changing the order of the df
    df.iloc[:,[0,1,2,4,3]].to_csv(args.output+'.txt', mode='a', sep="\t")
