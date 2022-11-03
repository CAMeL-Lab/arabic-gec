from camel_tools.utils.charsets import UNICODE_PUNCT_SYMBOL_CHARSET
import editdistance
import json
import string
import re
import argparse

"""A simple script post process the output for GEC.
   1) punctuation tokenize Arabic text
   2) compare the src to the prediction and ignore predictions
      that have a big edit distance compared to src.
      this deals with the issue of the M2 scorer being slow"""

def read_data_json(path):
    with open(path) as f:
        data = [json.loads(l) for l in f.readlines()]
    src = [x['gec']['raw'] for x in data]
    tgt = [x['gec']['cor'] for x in data]
    return src, tgt

def read_data(path):
    with open(path) as f:
        return [x.strip() for x in f.readlines()]

def postprocess(src, preds, output_path, verbose=False, gamma=100):
    puncs = string.punctuation + ''.join(list(UNICODE_PUNCT_SYMBOL_CHARSET))
    assert len(src) == len(preds)
    post_process_out = []
    skipped_sents = []

    with open(output_path, mode='w', encoding='utf8') as f:
        for i in range(len(src)):
            src_sent = src[i]
            pred_sent = preds[i]
            dist = editdistance.distance(src_sent, pred_sent)
            if dist >= gamma:
                post_process_out.append(src_sent)
                f.write(src_sent)
                skipped_sents.append((i, src_sent))
            else:
                pred_sent = re.sub(r'([' + re.escape(puncs) + '])(?!\d)', r' \1 ', pred_sent)
                pred_sent = re.sub(' +', ' ', pred_sent)
                pred_sent = pred_sent.strip()
                post_process_out.append(pred_sent)
                f.write(pred_sent)
            f.write('\n')

        print(f'{len(skipped_sents)} sentences were skipped')
        if verbose:
            for idx, sent in skipped_sents:
                print(f'{idx} :')
                print(f'SRC: {sent}')
                print(f'PRED: {preds[idx]}')
                print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( "--input_file_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir."
                        )
    parser.add_argument("--pred_file_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir."
                        )
    parser.add_argument("--output_file_dir",
                        default=None,
                        type=str,
                        help="The path of the output file"
                        )

    args = parser.parse_args()

    src, tgt = read_data_json(args.input_file_dir)
    preds = read_data(args.pred_file_dir)

    postprocess(src, preds, args.output_file_dir)
