from Levenshtein import editops
from getopt import getopt
import numpy as np
import codecs, sys


def _get_consecutive_ranges(list):
    sequences = np.split(list, np.array(np.where(np.diff(list) > 1)[0]) + 1)
    l = []
    for s in sequences:
        if len(s) > 1:
            l.append((np.min(s), np.max(s)))
        else:
            l.append(s[0])
    return l


def _read_align_file(input_file):
    f_content = codecs.open(input_file, "r", "utf8").read()
    f_sentences = f_content.split("\n\n")
    list_sentences = []
    for s in f_sentences:
        if len(s.split("\t")) > 1:
            sent_elt = []
            for l in s.split("\n"):
                sent_elt.append((l.split("\t")[0], l.split("\t")[2]))
            list_sentences.append(sent_elt)
    return list_sentences


def adjust_null_to_token(alignments):
    list_null_indexes = []
    i = 0
    for e in alignments:
        if e[0] == '':
            list_null_indexes.append(i)
        i += 1

    if len(list_null_indexes) > 0:
        ranges = _get_consecutive_ranges(list_null_indexes)
    else:
        ranges = []

    for rg in ranges:
        if type(rg) is tuple:
            al = alignments[rg[0]:rg[1] + 1]
            new_al = list(zip(*al))
            compressed_elt = " ".join(new_al[1])
            try:
                up_pair = (alignments[rg[0] - 1][0], alignments[rg[0] - 1][1] + " " + compressed_elt)
                down_pair = (alignments[rg[1] + 1][0], compressed_elt + " " + alignments[rg[1] + 1][1])

                if len(editops(up_pair[0], up_pair[1])) < len(editops(down_pair[0], down_pair[1])):
                    alignments[rg[0] - 1] = (alignments[rg[0] - 1][0], alignments[rg[0] - 1][1] + " " + compressed_elt)
                else:
                    alignments[rg[1] + 1] = (alignments[rg[1] + 1][0], compressed_elt + " " + alignments[rg[1] + 1][1])
            except:
                alignments[rg[0] - 1] = (alignments[rg[0] - 1][0], alignments[rg[0] - 1][1] + " " + compressed_elt)
        else:
            al = alignments[rg]
            try:
                up_pair = (alignments[rg - 1][0], alignments[rg - 1][1] + " " + al[1])
                down_pair = (alignments[rg + 1][0], al[1] + " " + alignments[rg + 1][1])

                if len(editops(up_pair[0], up_pair[1])) < len(editops(down_pair[0], down_pair[1])):
                    alignments[rg - 1] = (alignments[rg - 1][0], alignments[rg - 1][1] + " " + al[1])
                else:
                    alignments[rg + 1] = (alignments[rg + 1][0], al[1] + " " + alignments[rg + 1][1])
            except:
                alignments[rg - 1] = (alignments[rg - 1][0], alignments[rg - 1][1] + " " + al[1])

    new_alignments = []
    for al in alignments:
        if al[0] != '':
            new_alignments.append(al)
    return new_alignments


opts, args = getopt(sys.argv[1:], "v",
                    ["max_unchanged_words=", "beta=", "verbose", "ignore_whitespace_casing", "very_verbose", "output="])

if len(args) != 1:
    sys.exit(-1)

file = args[0]
sentences = _read_align_file(file)
aligned_sents = ''
for sentence in sentences:
    for e in adjust_null_to_token(sentence):
        aligned_sents += e[0] + "\t" + e[1] + "\n"
    aligned_sents += "\n"

sys.stdout.write(aligned_sents)
