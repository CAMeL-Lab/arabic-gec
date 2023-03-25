import Levenshtein
from typing import List
from camel_tools.utils.charsets import UNICODE_PUNCT_SYMBOL_CHARSET
import string


PUNCS = string.punctuation + ''.join(list(UNICODE_PUNCT_SYMBOL_CHARSET))


class Aligner:
    def _lcs(self, s, c):
        lcs = [[0] * (len(c) + 1) for _ in range(len(s) + 1)]
        step = [[0] * (len(c) + 1) for _ in range(len(s) + 1)]
        for i in reversed(range(len(s))):
            for j in reversed(range(len(c))):
                lcs[i][j], step[i][j] = lcs[i+1][j], 0

                for l in range(1, len(c)-j + 1):
                    if l > 3 * len(s[i]) + 8:
                        break

                    if c[j:j + l].isspace():
                        continue

                    if s[i] == c[j:j + l]:
                        weight = 1

                    elif s[i].strip() == c[j:j + l].strip():
                        weight = 0.75

                    else:

                        weight = Levenshtein.ratio(s[i], c[j:j + l]) * 0.5

                    if weight + lcs[i + 1][j + l] > lcs[i][j]:
                        lcs[i][j], step[i][j] = weight + lcs[i + 1][j + l], l

        return lcs, step


    def _rewrite_for_matching(self, s):
        return "".join(
            "." if c in PUNCS else c for c in s
        )

    def _best_alignment(self, s, c):
        lcs, step = self._lcs(list(map(self._rewrite_for_matching, s)), self._rewrite_for_matching(c))
        alignment = []

        j = 0
        for i in range(len(s)):
            l = len(c) - j if i + 1 == len(s) else step[i][j]
            alignment.append(c[j:j + l])
            j += l

        return alignment

    def align(self, subwords:List[str], chars:str) -> List[str]:
        return self._best_alignment(subwords, chars)