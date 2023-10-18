import codecs
import numpy as np
import sys
from getopt import getopt


def _generate_align_pairs(list_indices_input_correct, current_line):
    i = 0
    exp_indices = _expand_list_indices(list_indices_input_correct)
    align_pairs_list = []
    for w in current_line.split():
        if (i, i + 1) not in exp_indices:
            align_pairs_list.append((i, i + 1, w, w))
        i += 1

    align_pairs_list.extend(list_indices_input_correct)
    return sorted(align_pairs_list, key=lambda x: (x[0], x[1]))


def _expand_list_indices(list_indices_input_correct):
    expanded_list = []
    for e in list_indices_input_correct:
        if e[1] - e[0] > 1:
            intermediate_list = _gen_intermediate(e[0], e[1])
            expanded_list.extend(intermediate_list)
        else:
            expanded_list.append((e[0], e[1]))
    return expanded_list


def _gen_intermediate(first, last):
    new_l = []
    i = first
    while i < last:
        new_l.append((i, i + 1))
        i += 1
    return new_l


def _get_consecutive_ranges(list):
    sequences = np.split(list, np.array(np.where(np.diff(list) > 1)[0]) + 1)
    l = []
    for s in sequences:
        if len(s) > 1:
            l.append((np.min(s), np.max(s)))
        else:
            l.append(s[0])
    return l


def _reconstruct_raw_reference(list_indices_input_correct):
    new_sentence_raw = []
    new_sentence_reference = []
    for e in list_indices_input_correct:
        raw_input = " ".join(e[2].split())
        new_sentence_raw.append(raw_input)
        new_sentence_reference.append(e[3])
    return " ".join(new_sentence_raw), " ".join(new_sentence_reference)


def read_m2(ref_path):
    i = 0
    raw_sentences = ""

    with codecs.open(ref_path, "r", "utf8") as f:
        for l in f:
            if l == "\n":
                a = _generate_align_pairs(list_indices_input_correct, current_line)
                raw, ref = _reconstruct_raw_reference(a)
                raw_sentences = raw_sentences + " ".join(raw.split()) + "\n"
                continue
            if l[0] == "S":
                current_line = l
                list_indices_input_correct = []

            else:
                try:
                    begin = int(l.split("|||")[0].split()[1]) + 1
                    end = int(l.split("|||")[0].split()[2]) + 1
                    a = ' '.join(current_line.split()[begin:end])
                    b = l.split("|||")[2]
                    list_indices_input_correct.append((begin, end, a, b))
                    i += 1
                except:
                    print("error")
    return raw_sentences


def print_usage():
    print("Usage: generate-m2-source.py m2-file")
    print("where")
    print("  m2-file          -   the m2 file")


opts, args = getopt(sys.argv[1:], "v")

# starting point

if len(args) != 1:
    print_usage()
    sys.exit(-1)

m2_file = args[0]

sys.stdout.write(read_m2(m2_file))
