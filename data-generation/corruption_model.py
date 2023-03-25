from typing import List
import difflib
import functools
import json
import unicodedata
import copy
from collections import defaultdict, Counter
import re
import numpy as np


class CorruptModel:
    def __init__(self, model, examples, counts):
        self.model = model
        self.examples = examples
        self.counts = counts

    @classmethod
    def build(cls, data):
        model = defaultdict(lambda: defaultdict(lambda: 0))
        pruned_model = defaultdict(lambda: defaultdict(lambda: 0))

        counts = dict()
        pruned_counts = dict()

        examples = defaultdict(lambda: defaultdict(lambda: list()))


        for example in data:
            src_tokens, tgt_tokens, areta_tags, pos_tags = example
            assert len(src_tokens) == len(tgt_tokens) == len(areta_tags) == len(pos_tags)
            
            for i in range(len(src_tokens)):
                tgt_t, src_t, areta_tag, pos = tgt_tokens[i], src_tokens[i], areta_tags[i], pos_tags[i]

                if areta_tag != 'UC' and areta_tag != 'UNK':
                    rule = Rule.create(tgt_t, src_t, areta_tag)

                    if len(pos.split()) > 1: # we have to deal with this issue in the alignment
                        assert len(tgt_t.split()) > 1
                        if areta_tag == 'SPLIT':
                            model[(areta_tag, )][rule.to_str()] += 1
                            examples[(areta_tag, )][rule.to_str()].append({'correct': tgt_t, 'corrupted': src_t})
                            counts[(areta_tag, )] = 1 + counts.get((areta_tag, ), 0)
                    else:
                        model[(areta_tag, pos)][rule.to_str()] += 1
                        examples[(areta_tag, pos)][rule.to_str()].append({'correct': tgt_t, 'corrupted': src_t})
                        counts[(areta_tag, pos)] = 1 + counts.get((areta_tag, pos), 0)

        # prune out the rules at the tail
        for tag in model:
            for rule in model[tag]:
                if model[tag][rule] > 2:
                    pruned_model[tag][rule] = model[tag][rule]
                    pruned_counts[tag] = counts[tag]

        return cls(pruned_model, examples, pruned_counts)

    def __getitem__(self, tag):
        return self.model[tag]

    def __len__(self):
        return len(self.model)


class Rule:
    def __init__(self, patterns, ops):
        self.patterns = patterns
        self.ops = ops

    @classmethod
    def create(cls, src, tgt, areta_tag):
        patterns = []
        ops = []

        if areta_tag == 'SPLIT':
            words = src.split(' ')
            patterns.append((words[0], ))
            ops.append('merge')

        else:
            sm = difflib.SequenceMatcher(None,
                                         src,
                                         tgt,
                                         autojunk=False)

            for tag, i1, i2, j1, j2 in sm.get_opcodes():

                orig_i1 = i1
                orig_i2 = i2
                if orig_i1 == len(src):
                    orig_i1 -= 1

                if orig_i2 == len(src):
                    orig_i2 -= 1

                if i1 > (len(src) + 1) // 2:
                    i1 = i1 - len(src) - 1
                    i2 = i2 - len(src) - 1

                if tag != 'equal':
                    if tag == 'replace':
                        if i1 < 0:
                            s = i1 + len(src) + 1
                            e = i2 + len(src) + 1
                            patterns.append((i2, i1, src[s:e]))
                            ops.append((tag, i2, i1, tgt[j1:j2]))
                        else:
                            patterns.append((i1, i2, src[i1:i2]))
                            ops.append((tag, i1, i2, tgt[j1:j2]))

                    elif tag == 'delete':
                        if 'INSERT' not in areta_tag: # replace areta tags
                            if i1 < 0:
                                s = i1 + len(src) + 1
                                e = i2 + len(src) + 1
                                patterns.append((i2, i1, src[s:e]))
                                ops.append((tag, i2, i1, ''))
 
                            else:
                                patterns.append((i1, i2, src[i1:i2]))
                                ops.append((tag, i1, i2, ''))
                        else:
                            
                            if i1 < 0: # this case never happens
                                import pdb; pdb.set_trace()
                                patterns.append((src[i1+1:].strip(), ))
                                ops.append((tag, ))
                            else: # whole word delete
                                assert i1 == 0
                                patterns.append((i1, i2, src[i1:i2].strip(),))
                                ops.append((tag, i1, i1, ''))

                    elif tag == 'insert':
                        tag = 'add'
                        assert i1 == i2
                        if 'DELETE' in areta_tag:

                            patterns.append((None), )
                            ops.append((tag, tgt[j1: j2]))

                        elif 'MERGE' in areta_tag:

                            if i1 < 0:
                                assert i1 != -1
                                s = i1 + len(src) + 1
                                patterns.append((i1 - 1, i1, src[s:]))
                                ops.append((tag, i1 - 1, tgt[j1: j2]))
                            else:
                                patterns.append((0, i1, src[:i1]))
                                ops.append((tag, i1, tgt[j1: j2]))

                        else:
                            if i1 < 0:
                                s = len(src) + 1 + i1
                                patterns.append((i1 - 1, i1, src[s - 1: s]))
                                ops.append((tag, i1 - 1, tgt[j1: j2]))
                            else:

                                patterns.append((i1, i1 + 1, src[i1: i1 + 1]))
                                ops.append((tag, i1, tgt[j1: j2]))

        return cls(patterns, ops)

    def serialize(self):
        return [
            self.__class__.__name__,
            {field: value for field in self.serialize_fields() for value in [getattr(self, field)] if not isinstance(value, Rule)},
            {field: value.serialize() for field in self.serialize_fields() for value in [getattr(self, field)] if isinstance(value, Rule)},
        ]

    def serialize_fields(self):
        return ['patterns', 'ops']

    def to_str(self):
        return json.dumps(self.serialize(), ensure_ascii=False,
                          sort_keys=True, indent=None)

    def serialize_patterns(self):
        return [
            {field: value for field in ['patterns'] for value in [getattr(self, field)] if not isinstance(value, Rule)},
         ]

    def patterns_to_str(self):
        return json.dumps(self.serialize_patterns(), ensure_ascii=False,
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


    def is_applicable(self, text):

        applicable = [False] * len(self.patterns)

        for i, (pattern, op) in enumerate(zip(self.patterns, self.ops)):
            op_tag = op[0]

            if pattern == None: # Delete:
                return True

            if len(pattern) == 1: # SPLIT
                return text == pattern[0]

            s, e, seq = pattern

            s_idx = len(text) + s + 1 if s < 0 else s
            e_idx = len(text) + e + 1 if e < 0 else e

            s = e_idx if e_idx < s_idx else s_idx
            e = s_idx if s_idx > e_idx else e_idx

            applicable[i] = text[s: e] == seq

        return sum(applicable) == len(self.patterns)


    def apply(self, word):
        # this only applies to replace errors
        corr_word = list(word)
        ops = sorted(self.ops, key=lambda x: x[0], reverse=True)

        if len(self.patterns) == 1 and self.patterns[0] == None:
            return word + ' ' + ops[0][-1]

        if len(self.patterns) == 1 and self.ops[0] == 'merge':
            return None

        for op in ops:
            if op[0] == 'add':
                _, s, seq = op
                if s < 0:
                    s = len(corr_word) + 1 + s

                if s >= len(corr_word): return None
                corr_word[s], corr_word[s+1:] = seq, corr_word[s:]
            else:

                _, s, e, seq = op
                positive_s = (len(corr_word) + 1 + s) if s < 0 else s
                positive_e =  (len(corr_word) + 1 + e) if e < 0 else e

                s = positive_e if positive_e < positive_s else positive_s
                e = positive_s if positive_s > positive_e else positive_e

                if e > len(corr_word): return None

                corr_word[s: e] = seq

        return ''.join(corr_word)


class WordCorruptModel:
    def __init__(self, model):
        self.model = model

    @classmethod
    def build(cls, data):
        model = defaultdict(lambda: defaultdict(lambda: 0))

        counts = dict()

        for i, example in enumerate(data):

            src_tokens, tgt_tokens, areta_tags, pos_tags = example
            assert len(src_tokens) == len(tgt_tokens) == len(areta_tags) == len(pos_tags)

            for i in range(len(src_tokens)):
                tgt_t, src_t, areta_tag, pos = tgt_tokens[i], src_tokens[i], areta_tags[i], pos_tags[i]

                if areta_tag != 'UC' and areta_tag != 'UNK':

                    if len(pos.split()) > 1: # we have to deal with this issue in the alignment
                        assert len(tgt_t.split()) > 1
                        if areta_tag == 'SPLIT':
                            model[(areta_tag, '', tgt_t)][src_t] += 1
                            counts[(areta_tag, '', tgt_t)] = 1 + counts.get((areta_tag, '', tgt_t), 0)
                    else:
                        model[(areta_tag, pos, tgt_t)][src_t] += 1
                        counts[(areta_tag, pos, tgt_t)] = 1 + counts.get((areta_tag, pos, tgt_t), 0)

        # prune out the rules at the tail
        for tag in model:
            for src_t in model[tag]:
                model[tag][src_t] /= counts[tag]

        return cls(model)

    def __getitem__(self, tag):
        return self.model[tag]

    def __len__(self):
        return len(self.model)
