from .tokenizer import GECTokenizer
import difflib
from collections import defaultdict, Counter
import re
import Levenshtein
import json
from typing import List
from .subwords_aligner import Aligner
from .utils import Dataset
import random
import argparse

random.seed(42)

def read_data(path):
    ex = []
    all_examples = []

    with open(path) as f:
        for line in f.readlines()[1:]:
            line = line.replace('\n', '').split('\t')
            if len(line) > 1:
                src, tgt, tag = line
                ex.append((src, tgt, tag))
            else:
                all_examples.append(ex)
                ex = []

        if ex:
            all_examples.append(ex)

    return all_examples


class Rule:

    def __init__(self, src, tgt, negative_indices=True):
        self.edits = []

        if src != tgt and Levenshtein.ratio(src, tgt) < 0.3:
            self.edits.append(['replace_all', tgt])

        elif src != tgt:

            # we have to fix the spaces issue!
            # example of:
            # src: و اكثر
            # tgt: وأكثر
            sm = difflib.SequenceMatcher(a=src, b=tgt, autojunk=False, isjunk=None)

            for tag, i1, i2, j1, j2 in sm.get_opcodes():

                i1_src = i1
                i2_src = i2

                if negative_indices:
                    if i1 >= (len(src) + 1) // 2:
                        i1 = i1 - len(src) - 1
                        i2 = i2 - len(src) - 1

                if tag != 'equal':
                    if tag == 'delete':
                        if i1_src == 0 and i2_src == len(src):  # special case - delete whole word
                            edits.append(['delete_all'])

                        else:
                            self.edits.append([tag, i1, i2])

                    elif tag == 'insert':
                        self.edits.append([tag, i1, tgt[j1:j2]])

                    elif tag == 'replace':
                        if i1_src == 0 and i2_src == len(src):
                            self.edits.append(['replace_all', tgt[j1:j2]])

                        else:
                            self.edits.append([tag, i1, i2, tgt[j1:j2]])

                    else:
                        raise ValueError(f'Unknown value of tag {tag}')

        if len(self.edits) == 0:
            self.edits.append(['UC'])

        assert self.apply(src) == tgt


    def apply(self, text) -> str:


        text_splitted = list(text) + ['']  # last because of insert to the end of text instruction

        for edit, *args in self.edits:

            if edit == 'delete_all':
                return ''

            elif edit == 'replace_all':
                return args[0]

            elif edit == 'delete':
                if args[0] < -len(text_splitted): return None
                if args[1] > len(text_splitted): return None

                for i in range(args[0], args[1]):
                    text_splitted[i] = ''

            elif edit == 'insert':
                if args[0] >= len(text_splitted) or args[0] < -len(text_splitted): return None
                text_splitted[args[0]] = args[1] + text_splitted[args[0]]

            elif edit == 'replace':
                if args[0] < -len(text_splitted): return None
                if args[1] > len(text_splitted): return None

                for i in range(args[0], args[1]):
                    text_splitted[i] = ''

                text_splitted[args[0]] = args[2]

        return ''.join(text_splitted)

    def serialize(self):
        return [
            self.__class__.__name__,
            {field: value for field in self.serialize_fields() for value in [getattr(self, field)] if not isinstance(value, Rule)},
            {field: value.serialize() for field in self.serialize_fields() for value in [getattr(self, field)] if isinstance(value, Rule)},
        ]

    def serialize_fields(self):
        return ['edits']

    def to_str(self):
        return json.dumps(self.serialize(), ensure_ascii=False,
                          sort_keys=True, indent=None)


    @staticmethod
    def from_str(text):
        instance_name, primitives, rules = json.loads(text)
        instance_type = globals()[instance_name]
        instance = instance_type.__new__(instance_type)

        for key, value in primitives.items():
            setattr(instance, key, value)
        for key, value in rules.items():
            setattr(instance, key, Rule.from_str(json.dumps(value)))

        return instance



def create_rules(dataset, tokenizer, prune_rules=0):
    print(f'Creating rules...', flush=True)

    rules_dict = defaultdict(lambda: 0)
    wp_aligner = Aligner()

    # tokenizer = GECTokenizer('/scratch/ba63/BERT_models/bert-base-arabic-camelbert-msa')

    for example in dataset:
        src_tokens = example.src_tokens
        tgt_tokens  = example.tgt_tokens
        ged_tags =  example.ged_tags

        assert len(src_tokens) == len(tgt_tokens)

        for src, tgt in zip(src_tokens, tgt_tokens):
            # src_tokenized = _postprocess_bert_like(tokenizer.tokenize(src))
            src_tokenized = tokenizer._postprocess_bert_like(tokenizer.tokenize(src))

            alignment = wp_aligner.align(src_tokenized['clean'], tgt)

            # creating a rule for each word-piece
            for i, (src_wp, align) in enumerate(zip(src_tokenized['clean'], alignment)):
                rule = Rule(src_wp, align)

                rules_dict[rule.to_str()] += 1

    print(f'Done!', flush=True)

    sorted_rules = [(k, v) for k, v in  rules_dict.items()]
    sorted_rules = sorted(sorted_rules, key=lambda x: x[1], reverse=True)

    print(f'There are {len(sorted_rules)} unique rules!', flush=True)

    if prune_rules > 0:
        sorted_rules = [x for x in sorted_rules if x[1] >= prune_rules]
        print(f'There are {len(sorted_rules)} unique rules after pruning', flush=True)

    return sorted_rules


def annotate_data_with_rules(dataset, tokenizer, rules):
    wp_aligner = Aligner()
    # tokenizer = AutoTokenizer.from_pretrained('/scratch/ba63/BERT_models/bert-base-arabic-camelbert-msa')
    # tokenizer = GECTokenizer('/scratch/ba63/BERT_models/bert-base-arabic-camelbert-msa')
    annotated_data = []


    for example in dataset:
        annotated_example = []

        src_tokens = example.src_tokens
        tgt_tokens  = example.tgt_tokens
        ged_tags =  example.ged_tags


        assert len(src_tokens) == len(tgt_tokens)

        for src, tgt in zip(src_tokens, tgt_tokens):
            src_tokenized = tokenizer._postprocess_bert_like(tokenizer.tokenize(src))

            alignment = wp_aligner.align(src_tokenized['clean'], tgt)

            # creating a rule for each word-piece
            for i, (src_wp, align) in enumerate(zip(src_tokenized['clean'], alignment)):
                rule = Rule(src_wp, align)

                if rule.to_str() in rules:
                    annotated_example.append({'subword_model': src_tokenized['orig'][i],
                                              'subword': src_tokenized['clean'][i],
                                              'rule': rule.to_str()})

                else:
                    # if it's an OOV word, find the first rule that applies to it
                    found = False
                    shuffled_rules = list(rules)

                    random.seed(42)
                    random.shuffle(shuffled_rules)

                    for rule_str in shuffled_rules:
                        rule = Rule.from_str(rule_str)
                        if rule.apply(src_wp) == align:
                            found = True
                            annotated_example.append({'subword_model': src_tokenized['orig'][i],
                                                      'subword': src_tokenized['clean'][i],
                                                      'rule': rule.to_str()})
                            break

                    if found == False:
                        rule.edits = [['UNK']]
                        annotated_example.append({'subword_model': src_tokenized['orig'][i],
                                                  'subword': src_tokenized['clean'][i],
                                                  'rule': rule.to_str()})

        annotated_data.append(annotated_example)


    return annotated_data


def save_data(data, path, mode='train'):
    with open(path, 'w') as f:
        for ex in data:
            for x in ex:
                f.write(f"{x['subword_model']}\t{x['rule']}\n")
            f.write('\n')


    with open(path+'.json', 'w') as f:
        for ex in data:
            f.write(json.dumps(ex, ensure_ascii=False))
            f.write('\n')


def load_rules(path):
    with open(path) as f:
        return [x.strip() for x in f.readlines()]


def save_rules(rules, path):
    with open(path, 'w') as f:
        f.write('\n'.join(rules))
        f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alignment_file')
    parser.add_argument('--rules_file')
    parser.add_argument('--prune_rules', type=int, default=0)
    parser.add_argument('--mode')
    parser.add_argument('--output_file')

    args = parser.parse_args()

    tokenizer = GECTokenizer('/scratch/ba63/BERT_models/bert-base-arabic-camelbert-msa')
    dataset = Dataset(args.alignment_file)


    if args.mode == 'train':
        rules = create_rules(dataset, tokenizer, prune_rules=args.prune_rules)
        save_rules([x[0] for x in rules], args.rules_file)

        print(f'Annotating {args.mode} dataset using {len(rules)} rules..',
              flush=True)

        annotated_data = annotate_data_with_rules(dataset=dataset,
                                                tokenizer=tokenizer,
                                                rules=[x[0] for x in rules])


        save_data(annotated_data, args.output_file)

    else:
        rules = load_rules(args.rules_file)

        print(f'Annotating {args.mode} dataset using {len(rules)} rules..',
              flush=True)

        annotated_data = annotate_data_with_rules(dataset=dataset,
                                                  tokenizer=tokenizer,
                                                  rules=rules)

        save_data(annotated_data, args.output_file, mode=args.mode)
