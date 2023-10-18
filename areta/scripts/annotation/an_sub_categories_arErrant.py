from scripts.annotation.an_arabic_ops import remove_punctuation


def remove_tanween(word):
    new_s = []
    tanween_list = ["ً", "ٍ", "ٌ"]
    for c in word:
        if c not in tanween_list:
            new_s.append(c)
    return "".join(new_s)


def morph_error(all_paths, word, correct_word):
    path = all_paths[0]
    second_path = all_paths[1]
    list_gen = [{'enc0': ('3ms_dobj', '3fs_dobj')}, {'enc0': ('3ms_pron', '3fs_pron')},
                {'enc0': ('3fs_pron', '3ms_pron')}, {'enc0': ('3fs_dobj', '3ms_dobj')}]
    list_mt = [{'prc1': ('0', 'sa_fut')}, {'prc1': ('sa_fut', '0')}]
    list_xt = [{'enc0': ('3ms_pron', '0')}, {'prc1': ('bi_prep', '0')}, {'prc1': ('bi_part', '0')},
               {'prc1': ('li_prep', '0')}, {'enc0': ('1s_pron', '0')}, {'enc0': ('3fs_poss', '0')},
               {'enc0': ('1s_pron', '0')}]
    list_xm = [{'prc1': ('0', 'bi_part')}, {'prc1': ('0', 'bi_prep')}, {'enc0': ('3fs_dobj', '0')},
               {'enc0': ('0', '3fs_dobj')},
               {'enc0': ('0', '3ms_pron')}, {'enc0': ('1p_dobj', '0')}, {'enc0': ('0', '3fs_pron')},
               {'enc0': ('0', '3ms_poss')}, {'enc0': ('1s_dobj', '0')}, {'enc0': ('0', '1s_pron')},
               {'prc1': ('0', 'li_prep')}, {'enc0': ('0', '1s_poss')}, {'enc0': ('0', '3ms_dobj')}, ]
    list_xn = [{'enc0': ('3d_pron', '3mp_pron')}]
    err_sub_cat = []
    valid = False

    for p in all_paths:
        if len(p[0][1]) + len(p[0][0]) == 1:
            for e in p[0][1]:
                if "cas" in str(e) or "mod" in str(e):
                    err_sub_cat.append("XC")
                    valid = True
                    break

    if word + "اً" == correct_word or correct_word + "اً" == word:
        for e in second_path[0][1]:
            if "cas" in str(e) or "mod" in str(e):
                err_sub_cat.append("XC")
                valid = True
                break

    for e in path[0][1]:
        if "asp" in str(e) or e in list_mt:
            err_sub_cat.append("MT")
            valid = True
        if "cas" in str(e) or "mod" in str(e):
            err_sub_cat.append("XC")
            valid = True
        if "Al_det" in str(e):
            err_sub_cat.append("XF")
            valid = True
        if "gen" in str(e) or e in list_gen:
            err_sub_cat.append("XG")
            valid = True
        if "num" in str(e) or e in list_xn:
            err_sub_cat.append("XN")
            valid = True
        if e in list_xt:
            err_sub_cat.append("XT")
            valid = True
        # if e in list_xm:
        #     err_sub_cat.append("XM")
        #     valid = True
    if valid:
        return err_sub_cat
    return ["MI"]


def orth_error(a, b, path):
    err_sub_cat = []
    valid = False
    valid_hamza = False
    valid_ot = False
    hamza_list = ['replace: Alef-->Alef With Hamza Above',
                  'replace: Alef With Hamza Below-->Alef',
                  'replace: Alef With Hamza Above-->Alef With Madda Above',
                  'replace: Alef With Hamza Below-->Alef With Hamza Above',
                  'replace: Alef With Hamza Above-->Alef With Hamza Below',
                  'replace: Alef-->Alef With Hamza Below',
                  'replace: Alef With Hamza Above-->Alef',
                  'replace: Hamza-->Yeh With Hamza Above',
                  'replace: Alef With Madda Above-->Alef With Hamza Above',
                  'replace: Hamza-->Alef With Hamza Above',
                  'replace: Alef-->Alef With Madda Above',
                  'replace: Alef With Hamza Below-->Alef With Madda Above',
                  'insert: Waw', 'replace: Waw With Hamza Above-->Hamza',
                  'insert: Alef', 'replace: Alef With Hamza Above-->Hamza',
                  'replace: Alef-->Waw With Hamza Above',
                  'replace: Yeh With Hamza Above-->Hamza']
    teh_marbuta_list = ['replace: Heh-->Teh Marbuta',
                        'replace: Teh Marbuta-->Heh',
                        'replace: Teh Marbuta-->Teh',
                        'replace: Teh-->Teh Marbuta',
                        'replace: Heh-->Teh']
    oa_list = ['replace: Yeh-->Alef Maksura',
               'replace: Hamza-->Alef Maksura',
               'replace: Alef Maksura-->Yeh']

    for e in path[0][0]:

        valid_ow = False
        if a + "ا" == b:
            err_sub_cat.append("OW")
            valid_ow = True
            valid = True

        if e in hamza_list and not valid_ow:
            err_sub_cat.append("OH")
            valid = True
            valid_hamza = True

        if e in teh_marbuta_list:
            err_sub_cat.append("OT")
            valid = True
            valid_ot = True

        if e in oa_list:
            err_sub_cat.append("OA")
            valid = True

        if a == "وبقينا":
            print("ggg")
        if remove_tanween(a) == remove_tanween(b):
            err_sub_cat.append("ON")
            valid = True

        if b + "ا" == a or b + "و" == a or b + "ي" == a:
            err_sub_cat.append("OG")
            valid = True

        if "replace" in e and not valid_hamza and not valid_ot:
            err_sub_cat.append("OR")
            valid = True

        if "delete" in e and not valid_hamza and not valid_ot:
            err_sub_cat.append("OD")
            valid = True

        if "insert" in e and not valid_ow:
            err_sub_cat.append("OM")
            valid = True
    if valid:
        return err_sub_cat
    return ["OO"]


def semantic_error(a_m, path):
    list_sf_reduced = [{'prc2': ('fa_conj', '0')}, {'prc2': ('0', 'wa_part')}, {'prc2': ('wa_part', '0')},
                       {'prc2': ('0', 'fa_conj')}]
    if len(path[0][0]) + len(path[0][1]) > len(a_m) / 2:
        return "SW"

    if len(path[0][0]) + len(path[0][1]) == 1 and len(path[0][1]) == 1 and path[0][1][0] in list_sf_reduced:
        return "SF"

    return "SW"


def punct_error(word, correct_word):
    valid_punct = False
    punct_class = "PC"
    if remove_punctuation(word)[2] != remove_punctuation(correct_word)[2]:
        valid_punct = True
        punct_class = get_puct_subclass(remove_punctuation(word)[2], remove_punctuation(correct_word)[2])

    return valid_punct, punct_class


def get_punct_error(word, correct_word):
    punct_class = "PC"
    if remove_punctuation(word)[2] != remove_punctuation(correct_word)[2]:
        punct_class = get_puct_subclass(remove_punctuation(word)[2], remove_punctuation(correct_word)[2])
    else:
        return ""

    return punct_class


def get_puct_subclass(l_punct_w, l_punct_c):
    n_l_punct_w = []
    n_l_punct_c = []
    if len(l_punct_w) > 0 and len(l_punct_c) > 0 and (
            l_punct_w != l_punct_c):
        return "PC"
    for p in l_punct_w:
        if p not in l_punct_c:
            n_l_punct_w.append(p)
    for p in l_punct_c:
        if p not in l_punct_w:
            n_l_punct_c.append(p)
    if len(n_l_punct_w) == 0:
        return "PM"
    if len(n_l_punct_c) == 0:
        return "PT"

    return "PC"


def is_pt():
    return ""
