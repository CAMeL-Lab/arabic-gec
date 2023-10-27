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


def collate_alignments(path):
    example = []
    examples = []

    with open(path) as f:
        for line in f.readlines()[1:]:
            line = line.replace('\n', '').split('\t')
            if len(line) > 1:
                s, t, tag = line[0], line[1], line[2]
                if tag == 'DELETE_PM' or tag == 'DELETE_XM': # Singular DELETE tag
                    tag = 'DELETE'

                elif ('+' in tag and tag not in tag_combs) or (tag == 'UNK'):
                    tag = 'UNK'

                elif tag == 'REPLACE_PM' or tag == 'REPLACE_PC' or tag == 'REPLACE_PT':
                    tag = 'REPLACE_P'

                example.append((s, t, tag))
            else:
                examples.append(example)
                example = []

        if example:
            examples.append(example)

    return examples


def project_span(word, label):
    if 'MERGE' not in label:
        return word.split(), [label for _ in range(len(word.split()))]

    return word.split(), ['MERGE-B']+['MERGE-I' for _ in range(len(word.split()) - 1)]


def write_data(path, alignment):
    with open(path, mode='w') as f:
        for example in alignment:
            for ex in example:
                src_token, tgt_token, tag = ex

                if src_token == '' and tgt_token != '': # ignore insertions 
                    continue

                words, labels = project_span(src_token, tag)
                if len(words) > 1:
                    print(words, labels)

                for i, (word, label) in enumerate(zip(words, labels)):
                    f.write(f'{word}\t{label}\n')

            f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')

    args = parser.parse_args()

    areta_alignment = collate_alignments(args.input)
    write_data(args.output, areta_alignment)

