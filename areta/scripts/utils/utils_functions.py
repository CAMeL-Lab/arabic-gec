
def convert_subcat_dict_to_list(d, tag):
    new_l_v = []
    new_l_k = []
    for k, v in d.items():
        new_l_v.append(str(v))
        new_l_k.append(k + "/" + tag)
    return new_l_v, new_l_k

def convert_mapped_to_binary(tag):
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

def list_string_to_int_list(l):
    for i in range(0, len(l)):
        l[i] = int(l[i])
    return l

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

def get_score_orth(path):
    i = 0
    for e in path:
        i += 1
    return i


def get_score_morph(path):
    i = 0
    for e in path:
        i += 1
    return i

def get_sub_categories(err_faifi):
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
    # subcat_tags = config_data['subcat_tags_orig']
    for e in err_faifi.split("+"):
        subcat_tags[e.replace("\n", "")] = 1
    return subcat_tags

