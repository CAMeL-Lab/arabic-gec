import json
import re
import argparse

def read_data(path):
    with open(path, mode='r', encoding='utf8') as f:
        return [x.replace('\n', '') for x in f.readlines()]


def jsonify(src, tgt, all_tags, output_path):
    data = []
    for i in range(len(src)):
        s = src[i]
        t = tgt[i]
        if all_tags:
            tags = all_tags[i]
            assert len(tags.split()) == len(s.split())
            data.append({'raw': s, 'cor': t, 'ged_tags': tags})
        else:
            data.append({'raw': s, 'cor': t})

    with open(output_path, 'w') as f:
        for line in data:
            json.dump(line, f, ensure_ascii=False)
            f.write('\n')


def collate_ged_tags(ged_tags):
    src_ged = []
    all_src_ged = []
    for i, line in enumerate(ged_tags):
        line = line.strip()
        if line:
            src_token, tag = line.split('\t')
            src_ged.append({'src_token': src_token,
                            'tag': tag}
                            )
        else:
            all_src_ged.append(src_ged)
            src_ged = []

    return all_src_ged

def recover_src(src_sents, collated_data):
    x = []
    for ex in collated_data:
        x.append(' '.join([y['src_token'] for y in ex]))

    assert len(x) == len(src_sents)

    for i in range(len(x)):
        assert re.sub(' +', ' ', x[i]) == re.sub(' +', ' ', src_sents[i])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True, type=str, help='Source file.')
    parser.add_argument('--tgt', required=True, type=str, help='Target file.')
    parser.add_argument('--tags', default=None, type=str, help='GED tags file.')
    parser.add_argument('--output', required=True, type=str, help='Json output file')
    args = parser.parse_args()

    src = read_data(args.src)
    tgt = read_data(args.tgt)

    if args.tags:
        ged_tags = read_data(args.tags)
        all_src_ged = collate_ged_tags(ged_tags)

        # verify that we are able to recover the src sentences 
        # from the ged alignmnet file
        recover_src(src, all_src_ged)
        tags = [" ".join([x['tag'] for x in y]) for y in all_src_ged]
    else:
        tags = None

    jsonify(src=src, tgt=tgt, all_tags=tags,
            output_path=args.output)

