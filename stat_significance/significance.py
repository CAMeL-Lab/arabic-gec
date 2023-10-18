import random
import json
import argparse


def load_scores(path):
    with open(path) as f:
        return [json.loads(l) for l in f.readlines()]


def aggregate_score(scores):
    """
    Aggregates the m2scorer metrics by looking at the overall correct,
    proposed, and gold edits. This is the same as running the m2scorer over
    the entire output.

    Args:
        scores (`list` of `dict`): list of dicts of m2 metrics.

    Returns:
        dict: The aggregated metrics.
    """

    correct = sum([score['correct'] for score in scores])
    proposed = sum([score['proposed'] for score in scores])
    gold = sum([score['gold'] for score in scores])

    p = correct / proposed
    r = correct / gold
    f1 = 2.0 * p * r / (p + r)
    f05 = (1.0 + 0.5 * 0.5) * p * r / (0.5 * 0.5 * p + r)

    return {'p': p, 'r': r, 'f1': f1, 'f05': f05}


def paired_ar_test(system1_scores, system2_scores, n_trials=10000, seed=12345):
    """
    A paired two-sided approximate randomization test.
    Allows performing a paired two-sided approximate randomization
    test to assess the statistical significance of the difference in
    performance between two systems which are run and measured on the same
    corpus.

    https://aclanthology.org/W14-3333.pdf
    https://aclanthology.org/W05-0908.pdf
    https://aclanthology.org/P18-1128.pdf

    Args:
        system1_scores (list of dict): system1 m2 scores.
        system2_scores (list of dict): system1 m2 scores.
        n_trials (int, optional): number of shuffles. Defaults to 10000.
        seed (int, optional): Defaults to 12345.

    Returns:
        p_value: the p_value of the test
    """
    random.seed(seed)

    # absolute difference between system 1 and the system 2
    diff = abs(aggregate_score(system1_scores)['f05'] -
              aggregate_score(system2_scores)['f05'])

    # diff = abs(aggregate_score(system1_scores)['r'] -
    #            aggregate_score(system2_scores)['r'])
    c = 0

    # get shuffled pseudo systems
    for _ in range(n_trials):
        pseudo_system1_scores = []
        pseudo_system2_scores = []

        for score1, score2 in zip(system1_scores, system2_scores):
            if random.randint(0, 1) == 0:
                pseudo_system1_scores.append(score1)
                pseudo_system2_scores.append(score2)
            else:
                pseudo_system1_scores.append(score2)
                pseudo_system2_scores.append(score1)

        # aggregating scores
        pseudo_diff = abs(aggregate_score(pseudo_system1_scores)['f05'] -
                          aggregate_score(pseudo_system2_scores)['f05'])

        # pseudo_diff = abs(aggregate_score(pseudo_system1_scores)['r'] -
        #                   aggregate_score(pseudo_system2_scores)['r'])
        if pseudo_diff >= diff:
            c += 1

    p_value = (c + 1) / (n_trials + 1)

    return p_value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--system1_scores')
    parser.add_argument('--system2_scores')

    args = parser.parse_args()

    system1_scores = load_scores(args.system1_scores)
    system2_scores = load_scores(args.system2_scores)

    p_value = paired_ar_test(system1_scores, system2_scores)
    print(f'p-value: {p_value:.5f}')
