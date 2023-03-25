import json
import re
import argparse

tag_combs = [
        'REPLACE_OH+REPLACE_OM',
        'REPLACE_OH+REPLACE_OT',
        'REPLACE_OD+REPLACE_OR',
        'REPLACE_OD+REPLACE_OG',
        'REPLACE_XC+REPLACE_XN',
        'REPLACE_OA+REPLACE_OH',
        'REPLACE_OM+REPLACE_OR',
        'REPLACE_OH+REPLACE_XC',
        'REPLACE_OD+REPLACE_OH',
        'REPLACE_XC+REPLACE_XG',
        'REPLACE_MI+REPLACE_OH',
        'REPLACE_OA+REPLACE_OR',
        'REPLACE_OR+REPLACE_OT',
        'REPLACE_OD+REPLACE_OM'
        ]

def prep_data(path):
    src_sents = []
    tgt_sents = []

    src_sent = []
    tgt_sent = []


    with open(path) as f:
        for line in f.readlines()[1:]:
            line = line.replace('\n','')
            if line:
                src, tgt, tag = line.split('\t')

                if tag == 'UC':
                    src_sent.append(src)

                elif 'INSERT' not in tag:
                    if '+' in tag and tag not in tag_combs:
                        tag = 'I'

                    elif 'REPLACE' in tag:
                        tag = tag.replace('REPLACE', 'R')

                    elif 'DELETE' in tag:
                        tag = tag.replace('DELETE', 'D')

                    elif 'MERGE' in tag:
                        tag = tag.replace('MERGE', 'M')

                    elif 'SPLIT' in tag:
                        tag = 'SP'

                    src_sent.append(f'<{tag}>{src}</{tag}>')
                    tgt_sent.append(f'<{tag}>{tgt}</{tag}>')

            else:
                src_sents.append(' '.join(src_sent))
                tgt_sents.append(' '.join(tgt_sent))

                src_sent = []
                tgt_sent = []

    assert len(src_sents) == len(tgt_sents)

    return src_sents, tgt_sents



def jsonify(src_sents, tgt_sents, output_path):
    with open(output_path, mode='w') as f:
        for src_sent, tgt_sent in zip(src_sents, tgt_sents):
            f.write(json.dumps({'gec': {'src': src_sent, 'tgt': tgt_sent}},
                    ensure_ascii=False))
            f.write('\n')




if __name__ == '__main__':
    src_sents, tgt_sents = prep_data('/scratch/ba63/gec/data/alignment/modeling_areta_tags_improved/qalb14/qalb14_train.areta+.txt')
    jsonify(src_sents, tgt_sents, '/scratch/ba63/gec/data/bart-t5/qalb14/span/wo_camelira/train.json')

    src_sents, tgt_sents = prep_data('/scratch/ba63/gec/data/alignment/modeling_areta_tags_improved/qalb14/qalb14_tune.areta+.txt')
    jsonify(src_sents, tgt_sents, '/scratch/ba63/gec/data/bart-t5/qalb14/span/wo_camelira/tune.json')

    src_sents, tgt_sents = prep_data('/scratch/ba63/gec/data/gec_camelira/areta_tags/qalb14_train.areta+.txt')
    jsonify(src_sents, tgt_sents, '/scratch/ba63/gec/data/bart-t5/qalb14/span/w_camelira/train.json')

    src_sents, tgt_sents = prep_data('/scratch/ba63/gec/data/gec_camelira/areta_tags/qalb14_tune.areta+.txt')
    jsonify(src_sents, tgt_sents, '/scratch/ba63/gec/data/bart-t5/qalb14/span/w_camelira/tune.json')


    # parser = argparse.ArgumentParser()
    # parser.add_argument('--src', required=True, type=str, help='Source file.')
    # parser.add_argument('--tgt', required=True, type=str, help='Target file.')
    # parser.add_argument('--tags', default=None, type=str, help='GED tags file.')
    # parser.add_argument('--output', required=True, type=str, help='Json output file')
    # args = parser.parse_args()


