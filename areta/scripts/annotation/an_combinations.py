from Levenshtein import editops
from Levenshtein import apply_edit
from itertools import combinations
from scripts.annotation.an_compare_morph import morph_error_type
from scripts.annotation.an_compare_morph import morph_error_type_calimastar
from scripts.annotation.an_map_corr_tag import get_all_operations_labels


def _get_score(morph_changes):
    if len(morph_changes) == 0:
        return 1
    else:
        return 0.5


def _is_sublist_exist(full_list, l_target):
    for l in full_list:
        if l == l_target:
            return True
    return False


def get_error_annotation(w, wcorrect):
    input = editops(w, wcorrect)
    edit_combinations = sum([list(map(list, combinations(input, i))) for i in range(len(input) + 1)], [])

    posible_candidates = [{
        "spelling_": {
            "candidate": apply_edit(ed_comb, w, wcorrect), "operations": ed_comb,
            "operations_labels": get_all_operations_labels(ed_comb, w, wcorrect),
            "score": _get_score(morph_error_type(apply_edit(ed_comb, w, wcorrect), wcorrect)[1]),
        },

        "morph_error_type": morph_error_type(apply_edit(ed_comb, w, wcorrect), wcorrect)[0],
        "morph_features_change": morph_error_type(apply_edit(ed_comb, w, wcorrect), wcorrect)[1]} for
        ed_comb
        in
        edit_combinations]
    c_final = []
    for d in posible_candidates:
        if d["morph_error_type"] == "LEMMA_MATCH":
            c_final.append(d)
    return c_final


def get_error_annotation_calimastar(w, wcorrect):
    if w.startswith(" ") or w.endswith(" "):
        w = w.replace(" ", "")
    if wcorrect.startswith(" ") or wcorrect.endswith(" "):
        wcorrect = wcorrect.replace(" ", "")
    input = editops(w, wcorrect)
    edit_combinations = sum([list(map(list, combinations(input, i))) for i in range(len(input) + 1)], [])

    posible_candidates = [{
        "spelling_": {
            "candidate": apply_edit(ed_comb, w, wcorrect), "operations": ed_comb,
            "operations_labels": get_all_operations_labels(ed_comb, w, wcorrect),
            "score": _get_score(morph_error_type_calimastar(apply_edit(ed_comb, w, wcorrect), wcorrect)[1]),
        },

        "morph_error_type": morph_error_type_calimastar(apply_edit(ed_comb, w, wcorrect), wcorrect)[0],
        "morph_features_change": morph_error_type_calimastar(apply_edit(ed_comb, w, wcorrect), wcorrect)[1]} for
        ed_comb
        in
        edit_combinations]
    c_final = []
    for d in posible_candidates:
        if d["morph_error_type"] == "LEMMA_MATCH":
            for morph_sub_path in d["morph_features_change"]:
                new_d = d.copy()
                new_d["morph_features_change"] = morph_sub_path
                if _is_sublist_exist(c_final, new_d) == False:
                    c_final.append(new_d)
        if d["morph_error_type"] == "SAME_WORD":
            new_d = d.copy()
            new_d["morph_features_change"] = []
            if _is_sublist_exist(c_final, new_d) == False:
                c_final.append(new_d)
    return c_final


def get_correction_paths(err_tags):
    corr_paths = []
    for tag in err_tags:
        l = []
        l.append(tag["spelling_"]["operations_labels"])
        l.append(tag["morph_features_change"])
        corr_paths.append(l)
    return corr_paths
