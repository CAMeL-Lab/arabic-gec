from scripts.annotation.an_combinations import get_error_annotation_calimastar
from scripts.annotation.an_combinations import get_correction_paths
from scripts.annotation.an_arabic_ops import is_word_added, \
    is_word_deleted, is_number_converted, is_letters_swapped, remove_punctuation, is_punct_exist, \
    is_part_semantic, is_al_morph, is_confused_alif_ya
from scripts.annotation.an_sub_categories_arErrant import semantic_error, morph_error, orth_error, get_punct_error

path_option = "shortest_path"  # shortest_path or explainable_path

semantic_word_exception_list = ["لم", "لا", "ما", "من", "ماذا", "على", "إلى", "عن", "و", "أو", "لن", "لم", "له", "عليه"]


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


def _get_dominant_solution(paths):
    dominant = paths[0]
    for p in paths[1:]:
        if p[1] <= dominant[1] and p[2] >= dominant[2]:
            dominant = p
    return dominant


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


def _get_explainable_path(paths):
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


def remove_tanween(word):
    new_s = []
    tanween_list = ["ً", "ٍ", "ٌ"]
    for c in word:
        if c not in tanween_list:
            new_s.append(c)
    return "".join(new_s)


def explain_error(raw, correct):
    raw = " ".join(raw.split())
    correct = " ".join(correct.split())

    a = remove_punctuation(raw)[1]
    b = remove_punctuation(correct)[1]

    a = " ".join(a.split())
    b = " ".join(b.split())

    if raw == correct:
        # print("UNCHANGED")
        err_type_tag = "UC"

    elif a == b:
        err_type_tag = ""

    elif a.replace(" ", "") == b or a.replace("#", "") == b:
        err_type_tag = "SP"

    elif a == b.replace(" ", ""):
        err_type_tag = "MG"

    elif len(a) > 15 or len(b) > 15:
        err_type_tag = "UNK"

    elif is_word_deleted(a, b):
        # print("WORD_DELETED")
        err_type_tag = "XM"

    elif is_word_added(a, b):
        # print("WORD_ADDED")
        err_type_tag = "XT"

    elif is_al_morph(a, b):
        # print("MORPH_ERROR")
        err_type_tag = "XF"

    elif is_confused_alif_ya(a, b):
        # print("ORTH_ERROR")
        err_type_tag = "OA"

    elif is_number_converted(a, b):
        # print("CONVERTED_NUMBER/ORTH")
        err_type_tag = "OR"

    elif is_letters_swapped(a, b):
        # print("SWAPPED_LETTERS")
        err_type_tag = "OC"

    elif remove_tanween(a) == remove_tanween(b):
        err_type_tag = "ON"
        # print("ON")

    elif is_part_semantic(a, b) or (a in semantic_word_exception_list and b in semantic_word_exception_list):
        # print("SEMANTIC_ERROR")
        err_type_tag = "SW"

    else:
        errors = get_error_annotation_calimastar(a, b)
        if path_option == "shortest_path":
            all_paths = get_shortest_path(errors)
            path = all_paths[0]
            if len(all_paths) > 1:
                second_path = all_paths[1]
            else:
                second_path = None
        if path_option == "explainable_path":
            path = _get_explainable_path(errors)[0]
        if path_option == "optimised_unsup_path":
            path = _get_reranked_paths(errors)

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

        if (len(path[0][0]) + len(path[0][1]) > len(a) / 2 or (
                len(path[0][0]) + len(
            path[0][1]) == 1 and len(path[0][1]) == 1 and path[0][1][0] in list_sf)) and not (
                is_punct_exist(a) != is_punct_exist(b)):

            # print("SEMANTIC_ERROR")
            err_type_tag = semantic_error(a, path)

        elif (len(path[0][0]) + len(
                path[0][1]) == 1 and len(path[0][1]) == 1 and path[0][1][0] in list_mx):
            # print("WORD_DELETED")
            err_type_tag = "XM"

        elif (len(path[0][0]) + len(
                path[0][1]) == 1 and len(path[0][1]) == 1 and path[0][1][0] in list_xt):
            # print("WORD_ADDED")
            err_type_tag = "XT"

        elif len(path[0][0]) == 0:
            # print("MORPH_ERROR")
            is_xm_valid = False

            for e in path[0][1]:
                if e in list_mx:
                    is_xm_valid = True
                    break
            if is_xm_valid:
                err_type_tag = "XM" + "+" + "+".join(morph_error(all_paths, a, b))
            else:
                err_type_tag = "+".join(morph_error(all_paths, a, b))

        elif len(path[0][1]) == 0:
            # print("ORTH_ERROR")
            err_type_tag = "+".join(orth_error(a, b, path))

        else:
            # print("MORPH_ERROR+ORTH_ERROR")
            err_type_tag = "+".join(orth_error(a, b, path)) + "+" + "+".join(morph_error(all_paths, a, b))

    if get_punct_error(raw, correct) != "" and err_type_tag == "":
        err_type_tag = get_punct_error(raw, correct)

    elif get_punct_error(raw, correct) != "" and err_type_tag != "":
        err_type_tag = "+".join([err_type_tag, get_punct_error(raw, correct)])

    return err_type_tag

    # except Exception as ex:
    #
    #     # print("UNKNOWN_ERROR_TYPE")
    #     explain_errors = get_explained_error_subclass(a, b)
    #     fo = codecs.open("../../fout2.basic", "r", "utf8")
    #     fo.close()
    #     return "+".join(explain_errors)

