#!/usr/bin/env python3

# This file is part of the NUS M2 scorer.
# The NUS M2 scorer is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# The NUS M2 scorer is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# file: m2scorer.py
# 
# score a system's output against a gold reference 
#
# Usage: m2scorer.py [OPTIONS] proposed_sentences source_gold
# where
#  proposed_sentences   -   system output, sentence per line
#  source_gold          -   source sentences with gold token edits
# OPTIONS
#   -v    --verbose             -  print verbose output
#   --very_verbose              -  print lots of verbose output
#   --max_unchanged_words N     -  Maximum unchanged words when extracting edits. Default 2."
#   --beta B                    -  Beta value for F-measure. Default 0.5."
#   --ignore_whitespace_casing  -  Ignore edits that only affect whitespace and caseing. Default no."
#

from __future__ import print_function
import sys
from . import levenshtein
from getopt import getopt
from .util import paragraphs
from .util import smart_open
import signal

def load_annotation(gold_file):
    source_sentences = []
    gold_edits = []
    fgold = smart_open(gold_file)
    puffer = fgold.read()
    fgold.close()
    puffer = puffer.decode('utf8')
    for item in paragraphs(puffer.splitlines(True)):
        item = item.splitlines(False)
        sentence = [line[2:].strip() for line in item if line.startswith('S ')]
        assert sentence != []
        annotations = {}
        for line in item[1:]:
            if line.startswith('I ') or line.startswith('S '):
                continue
            assert line.startswith('A ')
            line = line[2:]
            fields = line.split('|||')
            start_offset = int(fields[0].split()[0])
            end_offset = int(fields[0].split()[1])
            etype = fields[1]
            if etype == 'noop':
                start_offset = -1
                end_offset = -1
            corrections =  [c.strip() if c != '-NONE-' else '' for c in fields[2].split('||')]
            # NOTE: start and end are *token* offsets
            original = ' '.join(' '.join(sentence).split()[start_offset:end_offset])
            annotator = int(fields[5])
            if annotator not in list(annotations.keys()):
                annotations[annotator] = []
            annotations[annotator].append((start_offset, end_offset, original, corrections))
        tok_offset = 0
        for this_sentence in sentence:
            tok_offset += len(this_sentence.split())
            source_sentences.append(this_sentence)
            this_edits = {}
            for annotator, annotation in annotations.items():
                this_edits[annotator] = [edit for edit in annotation if edit[0] <= tok_offset and edit[1] <= tok_offset and edit[0] >= 0 and edit[1] >= 0]
            if len(this_edits) == 0:
                this_edits[0] = []
            gold_edits.append(this_edits)
    return (source_sentences, gold_edits)


def evaluate(system_sentences_file, gold_file, max_unchanged_words=2,
             beta=1.0, ignore_whitespace_casing=False, verbose=False,
             very_verbose=False, timeout=None):

    # load source sentences and gold edits
    source_sentences, gold_edits = load_annotation(gold_file)

    # loading the system sentences
    with open(system_sentences_file) as f:
        system_sentences = [x.strip() for x in f.readlines()]

    p, r, f1, f05, skipped = levenshtein.batch_multi_pre_rec_f1(system_sentences, source_sentences,
                                                                gold_edits,
                                                                max_unchanged_words,
                                                                beta, ignore_whitespace_casing,
                                                                verbose, very_verbose, timeout)
    signal.alarm(0)


    m2_pp_sents = []

    if len(skipped) != 0:
        for i in range(len(system_sentences)):
            if i in skipped:
                m2_pp_sents.append(source_sentences[i])
            else:
                m2_pp_sents.append(system_sentences[i])

        with open(system_sentences_file+'.pp', "w", encoding="utf-8") as writer:
            writer.write("\n".join(m2_pp_sents))
            writer.write("\n")

    with open(system_sentences_file+'.m2', "w", encoding="utf-8") as writer:
            writer.write(f"Precision   : {p:.4f}\n")
            writer.write(f"Recall      : {r:.4f}\n")
            writer.write(f"F_1.0       : {f1:.4f}\n")
            writer.write(f"F_0.5       : {f05:.4f}\n")

    # return {'Precision': float(f'{p:.4f}'),
    #         'Recall': float(f'{r:.4f}'),
    #         'F1': float(f'{f1:.4f}'),
    #         'F0.5': float(f'{f05:.4f}'),
    #         'Skipped': skipped
    #         }


