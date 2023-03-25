import argparse
import glob
import os


def read_data(path):
    ex_pred =[] 
    ex_preds = []

    with open(path) as f:
        for line in f.readlines():
            line = line.strip()
            if line:
                ex_pred.append(line)
            else:
                ex_preds.append(ex_pred)
                ex_pred = []

        if ex_pred:
            ex_preds.append(ex_pred)
    
    return ex_preds


def postprocess(src_align_tags, pred_tags):
    """
    src_align_tags: the areta tags which are obtained from aligning
        the raw source with raw source after passing it to camelira.

    pred_tags: the predicted areta tags by the model that was trained on
        camelira's output.
    """

    # We have 4 cases to consider:
    # 1) UC and UC --> UC (correct uc prediction)
    # 2) UC and E --> E (model error)
    # 3) E and UC --> E (this is a camlira fix. We will rely on areta's
    #                    tag to compare to the model that was trained without camelira)
    # 4) E_x and E_y --> E_y (this is camlira fix and a model error)

    assert len(src_align_tags) == len(pred_tags)
    ex_preds = []

    for src_align, pred in zip(src_align_tags, pred_tags):
        assert len(src_align) == len(pred)
        ex_pred = []

        for x, y in zip(src_align, pred):
            if x == 'UC' and y == 'UC':
                pred_ = 'UC'

            elif x == 'UC' and y != 'UC':
                pred_ = y

            elif x != 'UC' and y == 'UC':
                pred_ = x

            elif x != 'UC' and y != 'UC':
                pred_ = y
            ex_pred.append(pred_)

        assert len(ex_pred) == len(src_align) == len(pred)
        ex_preds.append(ex_pred)


    assert len(ex_preds) == len(src_align_tags) == len(pred_tags)
    return ex_preds


def write(path, data):
    with open(path , mode='w') as f:
        for ex_preds in data:
            for x in ex_preds:
                f.write(x)
                f.write('\n')
            f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path')
    parser.add_argument('--alignment_tags')
    args = parser.parse_args()

    alignment_tags = read_data(args.alignment_tags)


    checkpoints = glob.glob(os.path.join(args.model_path, 'checkpoint-*/'))
    checkpoints += [args.model_path]

    for checkpoint in checkpoints:

        pred_tags = read_data(f'{checkpoint}/qalb14_tune.preds.txt')
        pred_tags_ = postprocess(alignment_tags, pred_tags)

        write(f'{checkpoint}/qalb14_tune.preds.pp.txt.checkme', pred_tags_)