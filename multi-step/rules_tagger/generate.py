import argparse
from .rules_factory.rules import Rule
import json
import re


def read_data(path):
    with open(path) as f:
        data = [json.loads(line) for line in f.readlines()]
    return data


def read_pred(path):
    ex_tags = []
    tags = []
    with open(path) as f:
        for line in f.readlines():
            line = line.strip()
            if line:
                ex_tags.append(line)
            else:
                tags.append(ex_tags)
                ex_tags = []

    return tags


def generate_example(tokens, rules):
    if len(tokens) != len(rules):
        import pdb; pdb.set_trace()
    assert len(tokens) == len(rules)
    generated = []

    for token, rule in zip(tokens, rules):
        rule = Rule.from_str(rule)

        if token['subword_model'].startswith('##'):
            generated.append(f"##{rule.apply(token['subword'])}")

        elif token['subword'].startswith(' '):
            generated.append(f" {rule.apply(token['subword'])}")

        else:
            generated.append(rule.apply(token['subword']))

    return generated


def generate(dataset, preds):
    assert len(dataset) == len(preds)
    generated = []

    for i in range(len(dataset)):
        gen = generate_example(dataset[i], preds[i])
        gen = detokenize(gen)
        generated.append(gen)
    
    return generated


def detokenize(seq):

    detokenize = []
    for token in seq:
        if token.startswith('##'):

            if len(detokenize) > 0:
                detokenize[-1] += token.replace('##','')
            else:
                detokenize.append(token.replace('##',''))


        elif token.startswith(' '):
            if len(detokenize) > 0:
                detokenize[-1] += token.strip()
            else:
                detokenize.append(token.strip())

        else:
            detokenize.append(token)

    detokenize = ' '.join(detokenize)
    detokenize = re.sub(' +', ' ', detokenize)

    return detokenize


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    parser.add_argument('--preds')
    parser.add_argument('--output')

    args = parser.parse_args()

    data = read_data(args.data)
    preds = read_pred(args.preds)

    generated = generate(data, preds)

    with open(args.output, 'w') as f:
        f.write('\n'.join(generated))
        f.write('\n')

