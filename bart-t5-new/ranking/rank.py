import json
import numpy as np
import argparse
import os
import glob


def read_data(path):
    with open(path) as f:
        return [x.strip() for x in f.readlines()]


def read_pred_tags(path):
    ex_tags = []
    sents_tags = []
    with open(path) as f:
        for line in f.readlines():
            line = line.strip()
            if line:
                line = line.split('\t')
                tag = line[-1]
                ex_tags.append(tag)
            else:
                sents_tags.append(ex_tags)
                ex_tags = []

        if ex_tags:
            sents_tags.append(ex_tags)

    return sents_tags


def read_gold_tags(path):
    with open(path) as f:
        data = [json.loads(line) for line in f.readlines()]
        tags = [x['gec']['ged_tags'] for x in data]
        tags = [x.split(' ') for x in tags]

    return tags


def hamming_distance(gold_tags, pred_tags):
    assert len(gold_tags) == len(pred_tags)

    return (np.array(gold_tags) != np.array(pred_tags)).sum()



def rank(nbest_hyps, nbest_ged_tags, gold_tags):
    assert len(nbest_hyps) == len(nbest_ged_tags)

    ranked_sents = []

    for sent_num in range(len(gold_tags)):
        sent_gold_tags = gold_tags[sent_num]
        nbest_sents = [x[sent_num] for x in nbest_hyps]
        nbest_sents_ged_tags = [x[sent_num] for x in nbest_ged_tags]

        hamming_dists = [hamming_distance(sent_gold_tags, ged_tags) for ged_tags in nbest_sents_ged_tags]

        min_dist_idx = np.argmin(hamming_dists)
        ranked_sents.append(nbest_sents[min_dist_idx])

    assert len(ranked_sents) == len(gold_tags)
    return ranked_sents


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_w_ged')
    parser.add_argument('--model_dir')

    args = parser.parse_args()
 
    gold_tags = read_gold_tags(args.src_w_ged)

    src_to_preds_ged_tags_files = glob.glob(f'{args.model_dir}/ranking/*areta+.txt.ged')

    nbest_ged_tags = []
    for file in src_to_preds_ged_tags_files:
        tags = read_pred_tags(file)
        nbest_ged_tags.append(tags)

    preds_files =  glob.glob(f'{args.model_dir}/qalb14_tune.preds.check.*.txt.pp')

    nbest_hyps = []
    for file in preds_files:
        preds = read_data(file)
        nbest_hyps.append(preds)

    ranked_hyps = rank(nbest_hyps, nbest_ged_tags, gold_tags)

    with open(f'{args.model_dir}/ranking/qalb14_tune.preds.ranked.txt', "w") as f:
        f.write("\n".join(ranked_hyps))
        f.write("\n")
