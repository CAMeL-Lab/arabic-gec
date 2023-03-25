from camel_tools.utils.charsets import UNICODE_PUNCT_SYMBOL_CHARSET
import editdistance
import json
import string
import re


"""A simple script post process the output for GEC.
   1) compare the src to the prediction and ignore predictions
      that have a big edit distance compared to src.
      this deals with the issue of the M2 scorer being slow"""

puncs = string.punctuation + ''.join(list(UNICODE_PUNCT_SYMBOL_CHARSET))

def read_data_json(path):
    with open(path) as f:
        data = [json.loads(l) for l in f.readlines()]
    src = [x['gec']['raw'] for x in data]
    return src


def read_data(path):
    with open(path) as f:
        return [x.strip() for x in f.readlines()]


def postprocess(src_path, preds_path, output_path, verbose=False, gamma=100):
    src = read_data_json(src_path)
    preds = read_data(preds_path)

    # pnx tokenize predictions
    preds = pnx_tokenize(preds)


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


def pnx_tokenize(data):
    pnx_re = re.compile(r'([' + re.escape(puncs) + '])(?!\d)')
    space_re = re.compile(' +')

    pnx_tokenized = []
    for line in data:
        line = line.strip()
        line = pnx_re.sub(r' \1 ', line)
        line = space_re.sub(' ', line)
        line = line.strip()
        pnx_tokenized.append(line)

    return pnx_tokenized