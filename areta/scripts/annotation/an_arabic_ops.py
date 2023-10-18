from Levenshtein import editops


def swap_letters(w, i):
    new_w = w[0:i] + w[i + 1] + w[i]
    if len(w) >= i + 2:
        new_w = new_w + w[i + 2:]
    return new_w


def get_swapped_set(w):
    swap_list = []
    for i in range(len(w) - 1):
        swap_list.append(swap_letters(w, i))
    return swap_list


def is_letters_swapped(word, correct_word):
    if correct_word in get_swapped_set(word):
        return True
    return False


def is_added_letter(word, correct_word):
    if len(editops(word, correct_word)) == 1 and editops(word, correct_word)[0][0] == "delete":
        return True
    return False


def is_missed_letter(word, correct_word):
    if len(editops(word, correct_word)) == 1 and editops(word, correct_word)[0][0] == "insert":
        return True
    return False


def is_confused_ha_ta(word, correct_word):
    list_ha_ta = [
        "ت",
        "ة",
        "ه"
    ]
    if len(editops(word, correct_word)) == 1 and editops(word, correct_word)[0][0] == "replace" and \
            (word[editops(word, correct_word)[0][1]] in list_ha_ta and correct_word[
                editops(word, correct_word)[0][2]] in list_ha_ta):
        return True
    return False


def is_confused_alif_ya(word, correct_word):
    list_alif_ya = [
        "ا",
        "ى",
        "ي"
    ]
    if len(word) > 1 and len(
            correct_word) > 1 and word[-1] in list_alif_ya and correct_word[-1] in list_alif_ya and word[
                                                                                                    0:-1] == correct_word[
                                                                                                             0:-1]:
        return True

    return False


def is_og(a, b):
    if b + "ا" == a or b + "و" == a or b + "ي" == a:
        return True

    return False


def is_al_morph(word, correct_word):
    if "ال" + word == correct_word:
        return True
    return False


def hamza_error(word, correct_word):
    list_hamz = ["ا",
                 "آ",
                 "ء",
                 "ئ",
                 "أ",
                 "ؤ",
                 "إ"]
    if len(editops(word, correct_word)) == 1 and editops(word, correct_word)[0][0] == "replace" and \
            word[editops(word, correct_word)[0][1]] in list_hamz and correct_word[
        editops(word, correct_word)[0][2]] in list_hamz:
        return True
    return False


def alif_fariqa(word, correct_word):
    if len(editops(word, correct_word)) == 1 and (
            word.endswith("و") and correct_word.endswith("وا") or word.endswith("وا") and correct_word.endswith("و")):
        return True
    return False


def og_rule(word, correct_word):
    if correct_word + "ا" == word or correct_word + "و" == word or correct_word + "ي" == word:
        return True
    return False


def is_part_semantic(s1, s2):
    part_list = ["عن", "لي", "ما", "مع", "بل", "على", "علي", "من", "في"]
    l1 = s1.split()
    l2 = s2.split()
    if len(l1) > 1:
        for e in l1:
            if e in part_list and s1.replace(e, "") == s2:
                return True
    if len(l2) > 1:
        for e in l2:
            if e in part_list and s2.replace(e, "") == s1:
                return True
    return False


def remove_punctuation(word):
    from unicodedata import category
    n_w = []
    punct_exist = False
    l_puncts = []
    for c in str(word):
        if category(c)[0] != 'P':
            n_w.append(c)
        else:
            l_puncts.append(c)
            punct_exist = True
    return punct_exist, "".join(n_w), "".join(l_puncts)


def is_punct_deleted(word, correct_word):
    from unicodedata import category
    word = word.replace("-", "")
    correct_word = correct_word.replace("-", "")
    if len(word) == 1 and category(word)[0] == 'P' and (
            correct_word == "" or str(correct_word) == "nan" or str(correct_word) == "Null"):
        return True

    # if remove_punctuation(word)[1] == remove_punctuation(correct_word)[1] and remove_punctuation(word)[
    #     0] == True and remove_punctuation(correct_word)[0] == False:
    #     return True

    n_l_punct_w = []
    n_l_punct_c = []
    l_punct_w = remove_punctuation(word)[2]
    l_punct_c = remove_punctuation(correct_word)[2]
    [n_l_punct_w.append(p) for p in l_punct_w if p not in l_punct_c]
    [n_l_punct_c.append(p) for p in l_punct_c if p not in l_punct_w]
    if len(n_l_punct_c) == 0 and remove_punctuation(word)[1] == remove_punctuation(correct_word)[1] and (
            remove_punctuation(correct_word)[
                0] == True or remove_punctuation(word)[
                0] == True):
        return True

    return False


def is_punct_exist(word):
    if remove_punctuation(word)[0]:
        return True
    return False


def is_punct_added(word, correct_word):
    from unicodedata import category
    word = word.replace("-", "")
    correct_word = correct_word.replace("-", "")
    if len(correct_word) == 1 and category(correct_word)[0] == 'P' and (
            word == "" or str(word) == "nan" or str(word) == "Null"):
        return True

    n_l_punct_w = []
    n_l_punct_c = []
    l_punct_w = remove_punctuation(word)[2]
    l_punct_c = remove_punctuation(correct_word)[2]
    [n_l_punct_w.append(p) for p in l_punct_w if p not in l_punct_c]
    [n_l_punct_c.append(p) for p in l_punct_c if p not in l_punct_w]
    if len(n_l_punct_w) == 0 and remove_punctuation(word)[1] == remove_punctuation(correct_word)[1] and (
            remove_punctuation(correct_word)[
                0] == True or remove_punctuation(word)[
                0] == True):
        return True

    return False


def punctuation_change(word, correct_word):
    from unicodedata import category
    word = word.replace("-", "")
    correct_word = correct_word.replace("-", "")
    if len(correct_word) == 1 and category(correct_word)[0] == 'P' and len(word) == 1 and category(word)[0] == 'P':
        return True
    if remove_punctuation(word)[1] == remove_punctuation(correct_word)[1] and remove_punctuation(word)[0] == True and \
            remove_punctuation(correct_word)[0] == True:
        return True
    return False


def is_word_deleted(word, correct_word):
    if (word == "" or str(word) == "nan" or str(word) == "Null") and is_punct_added(word, correct_word) == False:
        return True
    if len(correct_word.split()) == 2 and (correct_word.split()[0] == word or correct_word.split()[1] == word):
        return True
    if len(correct_word.split()) == 3 and (
            correct_word.split()[0] == word or correct_word.split()[1] == word or correct_word.split()[2] == word):
        return True
    return False


def is_word_added(word, correct_word):
    if (correct_word == "" or str(correct_word) == "nan" or str(correct_word) == "Null") and is_punct_deleted(word,
                                                                                                              correct_word) == False:
        return True
    return False


def is_number_converted(word, correct_word):
    import convert_numbers
    if convert_numbers.hindi_to_english(word) == correct_word:
        return True
    return False


def is_word_split(word, correct_word):
    if (len(word.split()) >= 2 or len(correct_word.split()) >= 2) and (
            "".join(word.split()) == "".join(correct_word.split())):
        return True
    return False
