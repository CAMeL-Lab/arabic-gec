import codecs, os
from Levenshtein import editops
import numpy as np

dirname = os.path.dirname(__file__)

input_path = os.path.join(dirname, "../../input/raw_qalb_test.txt")


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


def adjust_null_to_token(alignments):
    list_null_indexes = []
    i = 0
    for e in alignments:
        if e[0] == '':
            list_null_indexes.append(i)
        i += 1

    ranges = _get_consecutive_ranges(list_null_indexes)

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


def _read_align_file(input_file):
    list_pairs = []
    with codecs.open(input_file, "r", "utf8") as f:
        for l in f:
            # if l.strip():
            list_pairs.append((l.split("\t")[0].replace("\n", ""), l.split("\t")[1].replace("\n", "")))
            # else:
            #     list_pairs.append(("\n", "\n"))
    return list_pairs


def _reconstruct_raw_reference(list_indices_input_correct):
    new_sentence_raw = []
    new_sentence_reference = []
    for e in list_indices_input_correct:
        # raw_input = "#".join(e[2].split())
        raw_input = "<SSSS>".join(e[2].split())
        new_sentence_raw.append(raw_input)
        new_sentence_reference.append(e[3])
    return " ".join(new_sentence_raw), " ".join(new_sentence_reference)


def split_alignments_by_sentence(alignments):
    size = len(alignments)
    idx_list = [idx + 1 for idx, val in
                enumerate(alignments) if val == "\n"]

    res = [alignments[i: j] for i, j in
           zip([0] + idx_list, idx_list +
               ([size] if idx_list[-1] != size else []))]

    return res


def align_input_reference(ref_path, out_path):
    i = 0
    fw_raw = codecs.open(input_path, "w", "utf8")
    fw_align = codecs.open(out_path, "w", "utf8")
    with codecs.open(ref_path, "r", "utf8") as f:
        for l in f:
            if l == "\n":
                a = _generate_align_pairs(list_indices_input_correct, current_line)
                index = 0
                for e in a:
                    if index > 0:
                        fw_align.write("\t".join([e[2], e[3]]) + "\n")
                    index += 1
                # fw_align.write("\n")
                raw, ref = _reconstruct_raw_reference(a)
                fw_raw.write(" ".join(raw.split()[1:]) + "\n")
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

    fw_raw.close()
    fw_align.close()
    alignments = _read_align_file(out_path)
    alignments = adjust_null_to_token(alignments)
    fw_align = codecs.open(out_path, "w", "utf8")
    fw_align.write("source" + "\t" + "reference" + "\n")
    for al in alignments:
        # if al[0] == al[1] == '\n':
        #     fw_align.write("\n")
        # else:
        fw_align.write("\t".join(al) + "\n")
    fw_align.close()
