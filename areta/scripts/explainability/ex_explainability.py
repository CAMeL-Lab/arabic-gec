import codecs, os
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
import pandas as pd

from scripts.annotation.an_combinations import get_error_annotation_calimastar
from scripts.annotation.an_combinations import get_correction_paths
from scripts.annotation.an_arabic_ops import is_punct_added, is_punct_deleted, punctuation_change, is_word_added, \
    is_word_deleted, is_number_converted, is_letters_swapped, remove_punctuation, is_punct_exist, \
    is_part_semantic, is_al_morph, is_confused_alif_ya
from scripts.annotation.an_sub_categories_arErrant import punct_error, semantic_error, morph_error, orth_error

subcat_tags_orig = {"unchanged": 0,
                    "OH": 0,
                    "OT": 0,
                    "OA": 0,
                    "OW": 0,
                    "ON": 0,
                    # "OS": 0,
                    "OG": 0,
                    "OC": 0,
                    "OR": 0,
                    "OD": 0,
                    "OM": 0,
                    "OO": 0,
                    "MI": 0,
                    "MT": 0,
                    # "MO": 0,
                    "XC": 0,
                    "XF": 0,
                    "XG": 0,
                    "XN": 0,
                    "XT": 0,
                    "XM": 0,
                    "XO": 0,
                    "SW": 0,
                    "SF": 0,
                    # "SO": 0,
                    "PC": 0,
                    "PT": 0,
                    "PM": 0}
current_line = ""
nb_pairs = 0
unknown_tags = 0
nb_morph_spell = 0
nb_punct_added = 0
nb_punct_deleted = 0
nb_punct_changed = 0
nb_word_deleted = 0
nb_word_added = 0
nb_converted_number = 0
nb_spelling_error = 0
nb_word_split_error = 0
nb_letters_swapped_number = 0
nb_morph_errors = 0
nb_semantic_errors = 0
d_error = {}
all_pairs = []


def _convert_subcat_dict_to_list(d, tag):
    new_l_v = []
    new_l_k = []
    for k, v in d.items():
        new_l_v.append(str(v))
        new_l_k.append(k + "/" + tag)
    return new_l_v, new_l_k


path_option = "shortest_path"  # shortest_path or explainable_path
# path_option = "optimised_unsup_path"  # shortest_path or explainable_path

# fw = codecs.open("error_types_paths_faifi_new_manual.tsv", "w", "utf8")
#fw_sep = codecs.open("../../faifi_separate_classes_new_manual.tsv", "w", "utf8")

# d_map = {}
# dirname = os.path.dirname(__file__)
# file_map = os.path.join(dirname, "../../map.tsv")
#
# with codecs.open(file_map, "r", "utf8") as f:
#     for l in f:
#         d_map[l.split("\t")[0]] = l.split("\t")[1].replace("\n", "")

semantic_word_exception_list = ["لم", "لا", "ما", "من", "ماذا", "على", "إلى", "عن", "و", "أو", "لن", "لم", "له", "عليه"]

err_types = ["UNCHANGED/Predicted",
             "MORPH_ERROR/Predicted",
             "ORTH_ERROR/Predicted",
             "SEMANTIC_ERROR/Predicted",
             "PUNCTUATION_ERROR/Predicted",
             "WORD_ADDED/Predicted",
             "WORD_DELETED/Predicted"]

err_types_gold = ["UNCHANGED/Reference",
                  "MORPH_ERROR/Reference",
                  "ORTH_ERROR/Reference",
                  "SEMANTIC_ERROR/Reference",
                  "PUNCTUATION_ERROR/Reference",
                  "WORD_ADDED/Reference",
                  "WORD_DELETED/Reference"]

# fw_sep.write(
#     "RAW_WORD" + "\t" + "CORRECT_WORD" + "\t" + "ALFAFI_ERROR_TYPE" + "\t" + "\t".join(
#         err_types) + "\t" + "\t".join(
#         err_types_gold) + "\t" + "\t".join(_convert_subcat_dict_to_list(subcat_tags_orig, "Predicted")[
#                                                1]) + "\t" + "\t".join(
#         _convert_subcat_dict_to_list(subcat_tags_orig, "Reference")[
#             1]) + "\t" + "EDIT_TYPE" + "\t" + "ORTH_EDITS" + "\t" + "MORPH_EDITS" + "\n")


def get_shortest_path(paths):
    d = {}
    i = 0
    for path in get_correction_paths(paths):
        ops_zize = _get_score_orth(path[0]) + _get_score_morph(path[1])
        d[i] = (path, ops_zize)
        i = i + 1
    index_sort = sorted(d, key=lambda k: d[k][1])
    sorted_paths = []
    sorted_paths_t = []
    for e in index_sort:
        try:
            if len(d[e][0][0]) + len(d[e][0][1]) > 0:
                sorted_paths.append(d[e])
                # avg = get_average_score(d[e])
                # sorted_paths_t.append((d[e][0], d[e][1], avg))
        except:
            print("")
    # dominant = get_dominant_solution(sorted_paths_t)
    return sorted_paths


def _get_score_orth(path):
    i = 0
    for e in path:
        i += 1
    return i


def _get_score_morph(path):
    i = 0
    for e in path:
        i += 1
    return i


def _get_sub_categories(err_faifi):
    subcat_tags = {"unchanged": 0,
                   "OH": 0,
                   "OT": 0,
                   "OA": 0,
                   "OW": 0,
                   "ON": 0,
                   # "OS": 0,
                   "OG": 0,
                   "OC": 0,
                   "OR": 0,
                   "OD": 0,
                   "OM": 0,
                   "OO": 0,
                   "MI": 0,
                   "MT": 0,
                   # "MO": 0,
                   "XC": 0,
                   "XF": 0,
                   "XG": 0,
                   "XN": 0,
                   "XT": 0,
                   "XM": 0,
                   "XO": 0,
                   "SW": 0,
                   "SF": 0,
                   # "SO": 0,
                   "PC": 0,
                   "PT": 0,
                   "PM": 0,
                   }
    for e in err_faifi.split("+"):
        subcat_tags[e.replace("\n", "")] = 1
    return subcat_tags


def _get_dominant_solution(paths):
    dominant = paths[0]
    for p in paths[1:]:
        if p[1] <= dominant[1] and p[2] >= dominant[2]:
            dominant = p
    return dominant


# def get_average_score(path):
#     avg = 0
#     for e in path[0][0]:
#         if str(e) in d_count:
#             avg = avg + d_count[str(e)]
#         else:
#             avg = avg + 0
#     for e in path[0][1]:
#         if str(e) in d_count:
#             avg = avg + d_count[str(e)]
#         else:
#             avg = avg + 0
#     avg = avg / path[1]
#     return avg


def _get_reranked_paths(paths):
    d = {}
    i = 0
    for path in get_correction_paths(paths):
        ops_zize = len(path[0]) + len(path[1])
        d[i] = (path, ops_zize)
        i = i + 1
    index_sort = sorted(d, key=lambda k: d[k][1])
    sorted_paths = []
    # for e in index_sort:
    #     avg = get_average_score(d[e])
    #     sorted_paths.append((d[e][0], d[e][1], avg))
    dominant = _get_dominant_solution(sorted_paths)
    return dominant


def get_explainable_path(paths):
    d = {}
    i = 0
    for path in get_correction_paths(paths):
        ops_zize = len(path[0]) + len(path[1])
        d[i] = (path, len(path[1]) / ops_zize)
        i = i + 1
    index_sort = sorted(d, key=lambda k: d[k][1], reverse=True)
    sorted_paths = []
    for e in index_sort:
        sorted_paths.append(d[e])
    return sorted_paths


def get_edit_type(path):
    list_edits = []
    for e in path:
        if "insert:" in e:
            list_edits.append("ADD")
        if "replace:" in e:
            list_edits.append("SUB")
        if "delete" in e:
            list_edits.append("DEL")

    return ",".join(list(set(list_edits)))


def _map_score():
    d = {}
    mapped_err_type = []
    predicted_err_type = []
    list_labels_err_type = []
    with codecs.open("err_types_labels.txt", "r", "utf8") as f:
        for l in f:
            list_labels_err_type.append(l.replace("\n", ""))
    with codecs.open("../../map.tsv", "r", "utf8") as f:
        for l in f:
            d[l.split("\t")[0]] = l.split("\t")[1].replace("\n", "")
    i = 0
    with codecs.open("error_types_paths_faifi_new_manual.tsv", "r", "utf8") as f:
        for l in f:
            if i > 0:
                mapped_err_type.append(d[l.split("\t")[2]].replace("\n", ""))
                predicted_err_type.append(l.split("\t")[3].replace("\n", ""))
            i = i + 1
    f1_macro = f1_score(mapped_err_type, predicted_err_type, average='macro')
    f1_micro = f1_score(mapped_err_type, predicted_err_type, average='micro')
    acc_score = accuracy_score(mapped_err_type, predicted_err_type)

    print(f1_macro)
    print(f1_micro)
    print(acc_score)

    cm = confusion_matrix(mapped_err_type, predicted_err_type, labels=list_labels_err_type)
    cmd = ConfusionMatrixDisplay(cm, display_labels=list_labels_err_type)
    cmd.plot()

    cm_new = pd.DataFrame(cm, index=list_labels_err_type, columns=list_labels_err_type)
    print(classification_report(mapped_err_type, predicted_err_type))
    cm_new.to_csv('confusion_matrix_all_classes.csv')
    print("")


def list_string_to_int_list(l):
    for i in range(0, len(l)):
        l[i] = int(l[i])
    return l


def _map_score_multi_label():
    d = {}
    mapped_err_type = []
    predicted_err_type = []
    list_labels_err_type = []
    with codecs.open("err_types_labels.txt", "r", "utf8") as f:
        for l in f:
            list_labels_err_type.append(l.replace("\n", ""))
    with codecs.open("../../map.tsv", "r", "utf8") as f:
        for l in f:
            d[l.split("\t")[0]] = l.split("\t")[1].replace("\n", "")
    i = 0
    with codecs.open("../../faifi_separate_classes_new_manual.tsv", "r", "utf8") as f:
        for l in f:
            if i > 0:
                mapped_err_type.append(
                    list_string_to_int_list(_convert_mapped_to_binary(d[l.split("\t")[2]].replace("\n", ""))))
                predicted_err_type.append(list_string_to_int_list(",".join(l.split("\t")[3:10]).split(",")))
            i = i + 1
    mx = multilabel_confusion_matrix(mapped_err_type, predicted_err_type)
    f1_macro = f1_score(mapped_err_type, predicted_err_type, average='macro')
    f1_micro = f1_score(mapped_err_type, predicted_err_type, average='micro')
    acc_score = accuracy_score(mapped_err_type, predicted_err_type)

    print(f1_macro)
    print(f1_micro)
    print(acc_score)
    list_labels_err_type = ["UNCHANGED", "MORPH_ERROR", "ORTH_ERROR", "SEMANTIC_ERROR", "PUNCTUATION_ERROR",
                            "WORD_ADDED", "WORD_DELETED", "UNKNOWN_ERROR_TYPE"]
    # cm = confusion_matrix(mapped_err_type, predicted_err_type, labels=list_labels_err_type)
    # cmd = ConfusionMatrixDisplay(cm, display_labels=list_labels_err_type)
    # cmd.plot()

    # cm_new = pd.DataFrame(mx, index=list_labels_err_type, columns=list_labels_err_type)
    results = classification_report(mapped_err_type, predicted_err_type, output_dict=True)
    print(results)
    for k, v in results.items():
        line = [str(k)]
        for measure, val in v.items():
            if measure in ["precision", "recall", "f1-score", "support"]:
                # print(k + " " + str(measure) + " " + str(val))
                line.append(str(val))
        print("\t".join(line))
    # cm_new.to_csv('confusion_matrix_all_classes_multi_label.csv')
    print("")


def _map_score_multi_label():
    d = {}
    mapped_err_type = []
    predicted_err_type = []
    list_labels_err_type = []
    with codecs.open("err_types_labels.txt", "r", "utf8") as f:
        for l in f:
            list_labels_err_type.append(l.replace("\n", ""))
    with codecs.open("../../map.tsv", "r", "utf8") as f:
        for l in f:
            d[l.split("\t")[0]] = l.split("\t")[1].replace("\n", "")
    i = 0
    with codecs.open("../../faifi_separate_classes_new_manual.tsv", "r", "utf8") as f:
        for l in f:
            if i > 0:
                mapped_err_type.append(
                    list_string_to_int_list(_convert_mapped_to_binary(d[l.split("\t")[2]].replace("\n", ""))))
                predicted_err_type.append(list_string_to_int_list(",".join(l.split("\t")[3:10]).split(",")))
            i = i + 1
    mx = multilabel_confusion_matrix(mapped_err_type, predicted_err_type)
    f1_macro = f1_score(mapped_err_type, predicted_err_type, average='macro')
    f1_micro = f1_score(mapped_err_type, predicted_err_type, average='micro')
    acc_score = accuracy_score(mapped_err_type, predicted_err_type)

    print(f1_macro)
    print(f1_micro)
    print(acc_score)
    list_labels_err_type = ["UNCHANGED", "MORPH_ERROR", "ORTH_ERROR", "SEMANTIC_ERROR", "PUNCTUATION_ERROR",
                            "WORD_ADDED", "WORD_DELETED", "UNKNOWN_ERROR_TYPE"]
    # cm = confusion_matrix(mapped_err_type, predicted_err_type, labels=list_labels_err_type)
    # cmd = ConfusionMatrixDisplay(cm, display_labels=list_labels_err_type)
    # cmd.plot()

    # cm_new = pd.DataFrame(mx, index=list_labels_err_type, columns=list_labels_err_type)
    results = classification_report(mapped_err_type, predicted_err_type, output_dict=True)
    print(results)
    for k, v in results.items():
        line = [str(k)]
        for measure, val in v.items():
            if measure in ["precision", "recall", "f1-score", "support"]:
                # print(k + " " + str(measure) + " " + str(val))
                line.append(str(val))
        print("\t".join(line))
    # cm_new.to_csv('confusion_matrix_all_classes_multi_label.csv')
    print("")


def _multi_label_subclasses():
    d = {}
    mapped_err_type = []
    predicted_err_type = []

    with codecs.open("../../map.tsv", "r", "utf8") as f:
        for l in f:
            d[l.split("\t")[0]] = l.split("\t")[1].replace("\n", "")
    i = 0
    with codecs.open("../../faifi_separate_classes_new_manual.tsv", "r", "utf8") as f:
        for l in f:
            if i > 0:
                mapped_err_type.append(
                    list_string_to_int_list(list_string_to_int_list(",".join(l.split("\t")[43:69]).split(","))))
                predicted_err_type.append(list_string_to_int_list(",".join(l.split("\t")[17:43]).split(",")))
            i = i + 1
    mx = multilabel_confusion_matrix(mapped_err_type, predicted_err_type)
    f1_macro = f1_score(mapped_err_type, predicted_err_type, average='macro')
    f1_micro = f1_score(mapped_err_type, predicted_err_type, average='micro')
    acc_score = accuracy_score(mapped_err_type, predicted_err_type)

    print(f1_macro)
    print(f1_micro)
    print(acc_score)
    list_labels_err_type = ["UNCHANGED", "MORPH_ERROR", "ORTH_ERROR", "SEMANTIC_ERROR", "PUNCTUATION_ERROR",
                            "WORD_ADDED", "WORD_DELETED", "UNKNOWN_ERROR_TYPE"]
    # cm = confusion_matrix(mapped_err_type, predicted_err_type, labels=list_labels_err_type)
    # cmd = ConfusionMatrixDisplay(cm, display_labels=list_labels_err_type)
    # cmd.plot()

    # cm_new = pd.DataFrame(mx, index=list_labels_err_type, columns=list_labels_err_type)
    results = classification_report(mapped_err_type, predicted_err_type, output_dict=True)
    print(results)
    for k, v in results.items():
        line = [str(k)]
        for measure, val in v.items():
            if measure in ["precision", "recall", "f1-score", "support"]:
                # print(k + " " + str(measure) + " " + str(val))
                line.append(str(val))
        print("\t".join(line))
    # cm_new.to_csv('confusion_matrix_all_classes_multi_label.csv')
    print("")


def _convert_mapped_to_binary(tag):
    err_types = ["0",
                 "0",
                 "0",
                 "0",
                 "0",
                 "0",
                 "0",
                 "0"]
    valid = False
    if 'unchanged' in tag:
        err_types[0] = "1"
        valid = True

    if 'PUNCTUATION_MISSING' in tag:
        err_types[4] = "1"
        valid = True
    if 'PUNCTUATION_UNNECESSARY' in tag:
        err_types[4] = "1"
        valid = True
    if 'PUNCTUATION_CHANGED' in tag:
        err_types[4] = "1"
        valid = True
    if 'PUNCT_ERROR' in tag:
        err_types[4] = "1"
        valid = True
    if 'WORD_DELETED' in tag:
        err_types[6] = "1"
        valid = True
    if 'WORD_ADDED' in tag:
        err_types[5] = "1"
        valid = True
    if 'SEMANTIC_ERROR' in tag:
        err_types[3] = "1"
        valid = True
    if 'MORPH_ERROR' in tag:
        err_types[1] = "1"
        valid = True
    if 'ORTH_ERROR' in tag:
        err_types[2] = "1"
        valid = True
    # if valid == False:
    #     err_types = ["0",
    #                  "0",
    #                  "1",
    #                  "0",
    #                  "0",
    #                  "0",
    #                  "0",
    #                  "0"]

    return err_types[0:7]


def remove_tanween(word):
    new_s = []
    tanween_list = ["ً", "ٍ", "ٌ"]
    for c in word:
        if c not in tanween_list:
            new_s.append(c)
    return "".join(new_s)


def _wa_part_semantic(path):
    l = [{'prc2': ('0', 'wa_part')}]
    if len(path[0][0]) + len(path[0][1]) == 1 and len(path[0][1]) == 1 and path[0][1][0] in l:
        return True
    return False


def _map_score_err_detection():
    d = {}
    mapped_err_type = []
    predicted_err_type = []
    i = 0
    with codecs.open("error_types_paths_faifi_manual.tsv", "r", "utf8") as f:
        for l in f:
            if i > 0:
                if l.split("\t")[2].replace("\n", "") == 'unchanged':
                    mapped_err_type.append("no_error")
                else:
                    mapped_err_type.append("error")
                if l.split("\t")[3].replace("\n", "") == 'unchanged':
                    predicted_err_type.append("no_error")
                else:
                    predicted_err_type.append("error")
            i = i + 1
    f1_macro = f1_score(mapped_err_type, predicted_err_type, average='macro')
    f1_micro = f1_score(mapped_err_type, predicted_err_type, average='micro')
    acc_score = accuracy_score(mapped_err_type, predicted_err_type)

    print(f1_macro)
    print(f1_micro)
    print(acc_score)

    cm = confusion_matrix(mapped_err_type, predicted_err_type, labels=['no_error', 'error'])
    cmd = ConfusionMatrixDisplay(cm, display_labels=['no_error', 'error'])
    cmd.plot()

    cm_new = pd.DataFrame(cm, index=['no_error', 'error'], columns=['no_error', 'error'])
    print(classification_report(mapped_err_type, predicted_err_type, output_dict=True))
    cm_new.to_csv('confusion_matrix_err_detection.csv')
    print("")


# def get_super_class(tag):
#     if d_map[tag] == "unchanged":
#         return 0
#
#     if d_map[tag] == "MORPH_ERROR":
#         return 1
#
#     if d_map[tag] == "ORTH_ERROR":
#         return 2
#
#     if d_map[tag] == "SEMANTIC_ERROR":
#         return 3
#
#     if d_map[tag] == "PUNCTUATION_MISSING":
#         return 4
#
#     if d_map[tag] == "PUNCTUATION_UNNECESSARY":
#         return 4
#
#     if d_map[tag] == "PUNCTUATION_CHANGED":
#         return 4
#
#     if d_map[tag] == "WORD_ADDED":
#         return 5
#
#     if d_map[tag] == "WORD_DELETED":
#         return 6
#
#     return 0


def explain_error(raw, correct):
    try:
        subcat_tags = {"unchanged": 0,
                       "OH": 0,
                       "OT": 0,
                       "OA": 0,
                       "OW": 0,
                       "ON": 0,
                       # "OS": 0,
                       "OG": 0,
                       "OC": 0,
                       "OR": 0,
                       "OD": 0,
                       "OM": 0,
                       "OO": 0,
                       "MI": 0,
                       "MT": 0,
                       # "MO": 0,
                       "XC": 0,
                       "XF": 0,
                       "XG": 0,
                       "XN": 0,
                       "XT": 0,
                       "XM": 0,
                       "XO": 0,
                       "SW": 0,
                       "SF": 0,
                       # "SO": 0,
                       "PC": 0,
                       "PT": 0,
                       "PM": 0}

        a = raw
        b = correct

        if a == b:
            #print("UNCHANGED")
            return "uc"
        elif a.replace(" ", "") == b or a.replace("#", "") == b:
            return "MG"

        elif len(a) > 15 or len(b) > 15:
            return "UNK"

        elif is_punct_added(a, b):
            #print("PUNCTUATION_MISSING")
            return "PM"

        elif is_punct_deleted(a, b):
            #print("PUNCTUATION_UNNECESSARY")
            return "PT"

        elif punctuation_change(a, b):
            #print("PUNCTUATION_CHANGED")
            return "PC"

        elif is_word_deleted(a, b):
            #print("WORD_DELETED")
            return "XM"

        elif is_word_added(a, b):
            #print("WORD_ADDED")
            return "XT"

        elif is_al_morph(a, b):
            #print("MORPH_ERROR")
            return "XF"


        elif is_confused_alif_ya(a, b):
            #print("ORTH_ERROR")
            return "OA"


        elif is_number_converted(a, b):
            #print("CONVERTED_NUMBER/ORTH")
            return "OR"

        elif is_letters_swapped(a, b):
            #print("SWAPPED_LETTERS")
            return "OC"

        elif remove_tanween(a) == remove_tanween(b):
            subcat_tags["ON"] = 1
            #print("ON")
            return "ON"

        elif (is_part_semantic(a, b) or (
                remove_punctuation(a)[1] in semantic_word_exception_list and remove_punctuation(b)[
            1] in semantic_word_exception_list)) and not (
                is_punct_exist(a) != is_punct_exist(b)):
            #print("SEMANTIC_ERROR")
            #print("SW")
            return "SW"

        elif (is_part_semantic(a, b) or (
                remove_punctuation(a)[1] in semantic_word_exception_list and remove_punctuation(b)[
            1] in semantic_word_exception_list)) and (
                remove_punctuation(a)[2] != remove_punctuation(b)[2]):
            subcat_tags["SW"] = 1
            subcat_tags[punct_error(a, b)[1]] = 1
            return "SW" + "+" + "+".join(punct_error(a, b)[1])

        else:
            a_m = remove_punctuation(a)[1]
            b_m = remove_punctuation(b)[1]
            errors = get_error_annotation_calimastar(a_m, b_m)
            if path_option == "shortest_path":
                all_paths = get_shortest_path(errors)
                path = all_paths[0]
                if len(all_paths) > 1:
                    second_path = all_paths[1]
                else:
                    second_path = None
            if path_option == "explainable_path":
                path = get_explainable_path(errors)[0]
            if path_option == "optimised_unsup_path":
                path = _get_reranked_paths(errors)

            for e in path[0][0]:
                if e in d_error:
                    d_error[str(e)] = d_error[str(e)] + 1
                else:
                    d_error[str(e)] = 1

            for e in path[0][1]:
                new_e = str(list(e.keys())[0]) + str(list(e.values())[0])
                if new_e in d_error:
                    d_error[str(new_e)] = d_error[str(new_e)] + 1
                else:
                    d_error[str(new_e)] = 1

            list_sf = [{'prc2': ('fa_conj', '0')}, {'prc2': ('0', 'wa_part')}, {'prc2': ('wa_part', '0')},
                       {'prc2': ('0', 'fa_conj')}, {'prc1': ('bi_part', 'li_prep')},
                       {'prc1': ('bi_prep', 'li_prep')}, {'prc2': ('fa_conj', 'wa_part')},
                       {'prc1': ('bi_prep', 'li_prep')}, {'enc0': ('3ms_pron', '1s_pron')}, ]

            list_mx = [{'prc1': ('0', 'bi_part')}, {'prc1': ('0', 'bi_prep')}, {'enc0': ('3fs_dobj', '0')},
                       {'enc0': ('0', '3fs_dobj')},
                       {'enc0': ('0', '3ms_pron')}, {'enc0': ('1p_dobj', '0')}, {'enc0': ('0', '3fs_pron')},
                       {'enc0': ('0', '3ms_poss')}, {'enc0': ('1s_dobj', '0')}, {'enc0': ('0', '1s_pron')},
                       {'prc1': ('0', 'li_prep')}, {'enc0': ('0', '1s_poss')}, {'enc0': ('0', '3ms_dobj')}, ]

            list_xt = [{'enc0': ('3ms_pron', '0')}, {'prc1': ('bi_prep', '0')}, {'prc1': ('bi_part', '0')},
                       {'prc1': ('li_prep', '0')}, {'enc0': ('1s_pron', '0')}, {'enc0': ('3fs_poss', '0')},
                       {'enc0': ('1s_pron', '0')}]

            if (len(path[0][0]) + len(path[0][1]) > len(a_m) / 2 or (
                    len(path[0][0]) + len(
                path[0][1]) == 1 and len(path[0][1]) == 1 and path[0][1][0] in list_sf)) and not (
                    is_punct_exist(a) != is_punct_exist(b)):
                subcat_tags[semantic_error(a_m, path)] = 1
                #print("SEMANTIC_ERROR")
                return semantic_error(a_m, path)

            elif len(path[0][0]) + len(path[0][1]) > len(a_m) / 2 and (is_punct_exist(a) != is_punct_exist(b)):
                #print("SEMANTIC_ERROR+PUNCT_ERROR")
                subcat_tags[semantic_error(a_m, path)] = 1
                subcat_tags[punct_error(a, b)[1]] = 1
                return semantic_error(a_m, path) + "+" + punct_error(a, b)[1]

            elif (len(path[0][0]) + len(
                    path[0][1]) == 1 and len(path[0][1]) == 1 and path[0][1][0] in list_mx):
                #print("WORD_DELETED")
                subcat_tags["XM"] = 1
                return "XM"

            elif (len(path[0][0]) + len(
                    path[0][1]) == 1 and len(path[0][1]) == 1 and path[0][1][0] in list_xt):
                #print("WORD_ADDED")
                subcat_tags["XT"] = 1
                return "XT"

            elif len(path[0][0]) == 0 and not (remove_punctuation(a)[2] != remove_punctuation(b)[2]):
                #print("MORPH_ERROR")
                is_xm_valid = False
                for sub_cat in morph_error(all_paths, a_m, b_m):
                    subcat_tags[sub_cat] = 1
                err_types = ["0",
                             "1",
                             "0",
                             "0",
                             "0",
                             "0",
                             "0"]
                for e in path[0][1]:
                    if e in list_mx:
                        err_types = ["0",
                                     "1",
                                     "0",
                                     "0",
                                     "0",
                                     "0",
                                     "1"]
                        subcat_tags["XM"] = 1
                        is_xm_valid = True
                        break
                if is_xm_valid:
                    return "XM" + "+" + "+".join(morph_error(all_paths, a_m, b_m))
                else:
                    return "+".join(morph_error(all_paths, a_m, b_m))

            elif len(path[0][0]) == 0 and (remove_punctuation(a)[2] != remove_punctuation(b)[2]):
                #print("MORPH_ERROR+PUNCT_ERROR")
                for sub_cat in morph_error(all_paths, a_m, b_m):
                    subcat_tags[sub_cat] = 1
                subcat_tags[punct_error(a, b)[1]] = 1

                return "+".join(morph_error(all_paths, a_m, b_m))

            elif len(path[0][1]) == 0 and not (remove_punctuation(a)[2] != remove_punctuation(b)[2]):
                #print("ORTH_ERROR")
                for sub_cat in orth_error(a, b, path):
                    subcat_tags[sub_cat] = 1
                err_types = ["0",
                             "0",
                             "1",
                             "0",
                             "0",
                             "0",
                             "0"]
                return "+".join(orth_error(a, b, path))

            elif len(path[0][1]) == 0 and (remove_punctuation(a)[2] != remove_punctuation(b)[2]):
                #print("ORTH_ERROR+PUNCT_ERROR")
                for sub_cat in orth_error(a, b, path):
                    subcat_tags[sub_cat] = 1
                subcat_tags[punct_error(a, b)[1]] = 1
                return "+".join(orth_error(a, b, path))

            elif len(path[0][1]) > 0 and len(path[0][0]) > 0 and (
                    remove_punctuation(a)[2] != remove_punctuation(b)[2]):
                #print("MORPH_ERROR+ORTH_ERROR+PUNCT_ERROR")
                for sub_cat in orth_error(a, b, path):
                    subcat_tags[sub_cat] = 1
                for sub_cat in morph_error(all_paths, a_m, b_m):
                    subcat_tags[sub_cat] = 1
                subcat_tags[punct_error(a, b)[1]] = 1
                err_types = ["0",
                             "1",
                             "1",
                             "0",
                             "1",
                             "0",
                             "0"]
                return "+".join(orth_error(a, b, path)) + "+".join(morph_error(all_paths, a_m, b_m))
            else:
                #print("MORPH_ERROR+ORTH_ERROR")
                for sub_cat in orth_error(a, b, path):
                    subcat_tags[sub_cat] = 1
                for sub_cat in morph_error(all_paths, a_m, b_m):
                    subcat_tags[sub_cat] = 1
                err_types = ["0",
                             "1",
                             "1",
                             "0",
                             "0",
                             "0",
                             "0"]
                return "+".join(orth_error(a, b, path)) + "+" + "+".join(morph_error(all_paths, a_m, b_m))
    except Exception as ex:
        # print(ex)
        return 'UNK'

    print(
        "##############################################################################################################"
        "##################################################################################################################################")

# print(explain_error("", "كان"))
