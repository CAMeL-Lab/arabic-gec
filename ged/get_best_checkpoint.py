import os
import argparse
import glob

def get_best_checkpoint_ged(model_path):
    checkpoints = glob.glob(os.path.join(model_path, 'checkpoint-*/'))
    checkpoints += [model_path]
    checkpoint_scores = []

    for checkpoint in checkpoints:
        for eval_file in glob.glob(os.path.join(checkpoint, 'qalb14_dev.results.txt')):
            with open(eval_file) as f:
                metrics = [line.strip().replace(' ','').split(':')
                           for line in f.readlines()[:4]]
                metrics = {m[0]: m[1] for m in metrics}
                eval_metrics = metrics['f0.5']
                checkpoint_scores.append((eval_file, eval_metrics))

    return max(checkpoint_scores, key=lambda x: x[1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path')
    parser.add_argument('--task')
    args = parser.parse_args()

    best_checkpoint = get_best_checkpoint_ged(model_path=args.model_path)

    print(best_checkpoint, flush=True)
