import os
import argparse
import glob


def get_best_checkpoint_gec(model_path):
    checkpoints = glob.glob(os.path.join(model_path, 'checkpoint-*/'))
    checkpoints += [model_path]
    checkpoint_scores = []

    for checkpoint in checkpoints:
        for eval_file in glob.glob(os.path.join(checkpoint, 'zaebuc_dev.preds.oracle.txt.m2')):
            with open(eval_file) as f:
                f_score = f.readlines()[3].strip().split()[-1]
                checkpoint_scores.append((eval_file, f_score))

    return max(checkpoint_scores, key=lambda x: x[1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path')
    parser.add_argument('--task')
    args = parser.parse_args()
    best_checkpoint = get_best_checkpoint_gec(model_path=args.model_path)

    print(best_checkpoint, flush=True)
