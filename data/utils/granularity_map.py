import argparse

def map_data(path, mode='binary'):
    example = []
    examples = []

    with open(path) as f:
        for line in f.readlines():
            line = line.strip()
            if line:
                word, label = line.split('\t')
                if label != 'UC':
                    if mode == 'binary':
                        example.append((word, 'I'))

                    elif mode == 'coarse':
                        if (label == 'DELETE' or label == 'UNK'
                            or label == 'SPLIT' or 'MERGE' in label):
                            example.append((word, label))
                        else:
                            r_label = label.split('+')
                            r_label = [l.replace('REPLACE_', '')[0] for l in r_label]
                            r_label = [f'REPLACE_{l}' for l in r_label]
                            r_label = '+'.join(sorted(list(set(r_label))))
                            example.append((word, r_label))
                else:
                    example.append((word, label))

            else:
                examples.append(example)
                example = []

        if example:
            examples.append(example)

    return examples


def write_data(data, path):
    with open(path, mode='w') as f:
        for example in data:
            for token, label in example:
                f.write(f'{token}\t{label}')
                f.write('\n')
            f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--mode')
    parser.add_argument('--output')

    args = parser.parse_args()

    mapped_data = map_data(args.input, mode=args.mode)

    write_data(mapped_data, args.output)

