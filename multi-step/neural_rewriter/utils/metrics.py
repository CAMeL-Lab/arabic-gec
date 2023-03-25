import argparse

def accuracy(trg, pred):
     """Computes accuracy between a target sequence
     and a predicted sequence

     Args:
        - trg (str): reference
        - pred (str): generated output

     Returns:
        - acc (float): word level accuracy
     """
     trg_words = trg.split(' ')
     pred_words = pred.split(' ')
     acc = 0
     for i, w in enumerate(trg_words):
         if i < len(pred_words):
             if w == pred_words[i]:
                 acc += 1
         else:
             break
     return float(acc) / float(len(trg_words))

def accuracy_top_n(trg, top_n_preds):
    if trg in top_n_preds:
        return 1
    return 0

def corpus_accuracy(trg_corpus, pred_corpus):
    """Computes accuracy between a list of target
    sequences and a list of predicted sequences

    Args:
        - trg_corpus (list): list of references
        - pred_corpus (list): list of model's predictions

    Returns:
        - corpus_acc (float): average accuracy accross the corpus
    """
    corpus_acc = 0
    for i, line in enumerate(trg_corpus):
        corpus_acc += accuracy(trg=trg_corpus[i], pred=pred_corpus[i])
    return corpus_acc / len(trg_corpus)

def abs_length_diff(trg, pred):
    """Computes absolute length difference
    between a target sequence and a predicted sequence

    Args:
        - trg (str): reference
        - pred (str): generated output

    Returns:
        - absolute length difference (int)
     """
    trg_length = len(trg.split(' '))
    pred_length = len(pred.split(' '))
    return abs(trg_length - pred_length)

def corpus_abs_length_diff(trg_corpus, pred_corpus):
    """Computes abslute length difference
    between a list of target sequences and list
    of predicted sequences

    Args:
        - trg_corpus (list): list of references
        - pred_corpus (list): list of model's predictions
    Returns:
        - average absolute length diff accross the corpus (float)
    """
    abs_diff = 0
    for i, line in enumerate(trg_corpus):
        abs_diff += abs_length_diff(trg=trg_corpus[i], pred=pred_corpus[i])
    return float(abs_diff) / len(trg_corpus)

def read_examples(data_dir):
    with open(data_dir, mode='r', encoding='utf8') as f:
        return f.readlines()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--trg_directory',
        default=None,
        type=str,
        help="Directory of the target corpus"
    )

    parser.add_argument(
        '--pred_directory',
        default=None,
        type=str,
        help="Directory of the prediction corpus"
    )

    args = parser.parse_args()

    trg_examples = read_examples(args.trg_directory)
    pred_examples = read_examples(args.pred_directory)

    assert len(trg_examples) == len(pred_examples)

    accuracy = corpus_accuracy(trg_corpus=trg_examples,
                               pred_corpus=pred_examples)

    abs_diff = corpus_abs_length_diff(trg_corpus=trg_examples,
                                      pred_corpus=pred_examples)

    eval_res = "Accuracy{:>6s}{}\nAbs Length{:>4s}{}".format(": ", accuracy, ": ", abs_diff)
    print(eval_res)

if __name__ == "__main__":
    main()
