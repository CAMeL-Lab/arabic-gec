from camel_tools.utils.charsets import UNICODE_PUNCT_SYMBOL_CHARSET, AR_LETTERS_CHARSET
from camel_tools.utils.normalize import (
    normalize_alef_ar,
    normalize_alef_maksura_ar,
    normalize_teh_marbuta_ar
)
import string
import re


PUNCS = string.punctuation + ''.join(list(UNICODE_PUNCT_SYMBOL_CHARSET))
DIGIT_MAPPER={'٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4', '٥': '5',
              '٦': '6', '٧': '7', '٨': '8', '٩': '9'}

def norm_digits(string):
    s = ""
    if bool(re.search(r'\d', string)):
        for c in string:
            s += DIGIT_MAPPER.get(c, c)
    return s if s else string


def remove_kashida(string):
    # removes kashida if it exists in words
    if bool(re.search(r'', string)):
        words = string.split()
        clean_words = []
        for word in words:
            if len(word) > 1:
                clean_words.append(re.sub('\u0640', '', word))
            else:
                clean_words.append(word)
    s = " ".join(clean_words)
    return s if s else string


def norm_pnx_nums(string):
    norm_string = re.sub(r'(['+re.escape(PUNCS)+'])', r'PNX\1', string.strip())
    # Normalizing the numbers
    norm_string = norm_digits(norm_string)
    # Removing kashida
    norm_string = remove_kashida(norm_string)
    return norm_string


def norm_alef_ya_teh(string):
    norm_s = normalize_alef_ar(string)
    norm_s = normalize_alef_maksura_ar(norm_s)
    norm_s = normalize_teh_marbuta_ar(norm_s)
    return norm_s

