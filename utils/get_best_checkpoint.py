import os
import argparse

def get_best_checkpoint(model_path):
    checkpoints = os.listdir(model_path)

    checkpoint_scores = []
    for checkpoint in checkpoints:
        checkpoint_path = os.path.join(model_path, checkpoint)

        if 'checkpoint' in checkpoint_path:
            with open(os.path.join(checkpoint_path, 'm2.dev.qalb15.pred.txt.pnx.edits.eval')) as f:
                f_score = f.readlines()[3].strip().split()[-1]
                checkpoint_scores.append((checkpoint, f_score))

    # evaluating the last checkpoint
    with open(os.path.join(model_path, 'm2.dev.qalb15.pred.txt.pnx.edits.eval')) as f:
            f_score = f.readlines()[3].strip().split()[-1]
            checkpoint_scores.append((model_path, f_score))

    return max(checkpoint_scores, key=lambda x: x[1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path')
    args = parser.parse_args()
    best_checkpoint = get_best_checkpoint(model_path=args.model_path)

    print(best_checkpoint, flush=True)
