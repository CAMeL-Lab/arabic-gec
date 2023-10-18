from aligner.align_text_api import align_api
from Levenshtein import editops
import codecs, os
import numpy as np


def _prepare_alignments(alignments):
    new_alignments = []
    for al in alignments:
        if al != "\n":
            new_alignments.append((al.split("\t")[0], al.split("\t")[2]))
        # else:
            # new_alignments.append(("NEW LINE", "NEW LINE"))
    return new_alignments


def align_input_system(f_system_path, out_path):
    dirname = os.path.dirname(__file__)

    f_input_path = os.path.join(dirname, "../../input/raw_qalb_test.txt")
    raw_lines = codecs.open(f_input_path, "r", "utf8").readlines()
    system_lines = []
    with codecs.open(f_system_path, "r", "utf8") as f:
        for l in f:
            system_lines.append(" ".join(l.split()[1:]))

    alignments = align_api(raw_lines, system_lines)
    alignments = _prepare_alignments(alignments)
    alignments = adjust_null_to_token(alignments)
    fw = codecs.open(out_path, "w", "utf8")
    fw.write("source" + "\t" + "reference" + "\n")

    for al in alignments:
        # if "NEW LINE" in al[0] and "NEW LINE" in al[1]:
        #     fw.write("\t".join([al[0].replace("NEW LINE", ""), al[1].replace("NEW LINE", "")]) + "\n")
        # else:
            # fw.write("\t".join([al[0].replace("#", " "), al[1].replace("#", " ")]) + "\n")
        fw.write("\t".join([al[0].replace("<SSSS>", " "), al[1].replace("<SSSS>", " ")]) + "\n")


def align_ref_system_basic(f_system_path, f_ref_path, out_path):
    ref_lines = []
    with codecs.open(f_ref_path, "r", "utf8") as f:
        for l in f:
            ref_lines.append(" ".join(l.split()[1:]))
    system_lines = []
    with codecs.open(f_system_path, "r", "utf8") as f:
        for l in f:
            system_lines.append(" ".join(l.split()[1:]))

    alignments = align_api(system_lines, ref_lines)
    alignments_sent_by_sent = split_alignments_by_sentence(alignments)

    alignments = []
    for sub_al in alignments_sent_by_sent:
        sub_al = _prepare_alignments(sub_al)
        sub_al = adjust_null_to_token(sub_al)
        alignments.append(sub_al)

    fw = codecs.open(out_path, "w", "utf8")
    fw.write("source" + "\t" + "reference" + "\n")

    for sub_al in alignments:
        for al in sub_al:
            fw.write("\t".join([al[0].replace("<SSSS>", " "), al[1].replace("<SSSS>", " ")]) + "\n")
            # fw.write("\t".join([al[0], al[1]]) + "\n")
        fw.write("\n")


def split_alignments_by_sentence(alignments):
    size = len(alignments)
    idx_list = [idx + 1 for idx, val in
                enumerate(alignments) if val == "\n"]

    res = [alignments[i: j] for i, j in
           zip([0] + idx_list, idx_list +
               ([size] if idx_list[-1] != size else []))]

    return res


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
