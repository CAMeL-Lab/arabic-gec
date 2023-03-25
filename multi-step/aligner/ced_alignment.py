import sys
import math
from collections import deque
import editdistance
from aligner.utils import norm_pnx_nums


def _print_table(tbl, m, n):
    for i in range(0, m + 1):
        for j in range(0, n + 1):
            sys.stdout.write("%s/%s" % tbl[(i, j)])
            sys.stdout.write('\t')
        sys.stdout.write('\n')


def _edit_distance(tokens1, tokens2, weight_fns):
    tbl = {}
    tbl[(0, 0)] = (0, 'n')

    m = len(tokens1)
    n = len(tokens2)

    for i in range(0, m):
        tbl[(i + 1, 0)] = (i + 1, 'd')

    for j in range(0, n):
        tbl[(0, j + 1)] = (j + 1, 'i')

    if m == 0 or n == 0:
        return tbl

    for i in range(0, m):
        for j in range(0, n):
            if (tokens1[i] == tokens2[j]):
                edit_cost = tbl[(i + 1, j + 1)] = (tbl[(i, j)][0], 'n')
            else:
                edit_cost = (tbl[(i, j)][0] + weight_fns['s'](tokens1[i], tokens2[j]), 's')

            insert_cost = (tbl[(i, j + 1)][0] + weight_fns['d'](tokens1[i]), 'd')
            delete_cost = (tbl[(i + 1, j)][0] + weight_fns['i'](tokens2[j]), 'i')

            tbl[(i + 1, j + 1)] = min([insert_cost, delete_cost, edit_cost], key = lambda t: t[0])

    return tbl


def _gen_alignments(tokens1, tokens2, orig_tokens1, orig_tokens2):
    weight_fns = {
        's': lambda x, y: editdistance.eval(x, y) * 2 / max(len(x), len (y)),
        'd': lambda x: 1,
        'i': lambda x: 1
    }

    dist_table = _edit_distance(tokens1, tokens2, weight_fns)

    m = len(tokens1)
    n = len(tokens2)

    alignments = deque()

    i = m
    j = n

    while i != 0 or j != 0:
        op = dist_table[(i, j)][1]
        cost = dist_table[(i, j)][0]

        if op == 'n' or op == 's':
            _op = 'KEEP' if op == 'n' else 'REPLACE'

            orig_token1, orig_token2 = '', ''
            if orig_tokens1[i - 1] != tokens1[i - 1]:
                orig_token1 = orig_tokens1[i - 1]

            if orig_tokens2[j - 1] != tokens2[j - 1]:
                orig_token2 = orig_tokens2[j - 1]

            alignments.appendleft((i, j, tokens1[i - 1], tokens2[j - 1], _op, orig_token1,
                                   orig_token2))
            i -= 1
            j -= 1

        elif op == 'i':
            alignments.appendleft(('None', j, tokens2[j - 1], 'INSERT', '', orig_tokens2[j - 1]))
            j -= 1

        elif op == 'd':
            alignments.appendleft((i, 'None', tokens1[i - 1], 'DELETE', orig_tokens1[i - 1], ''))
            i -= 1

    return alignments


def align_words(s1, s2):
    norm_s1 = norm_pnx_nums(s1)
    norm_s2 = norm_pnx_nums(s2)

    s1_tokens = s1.split()
    s2_tokens = s2.split()

    norm_s1_tokens = norm_s1.split()
    norm_s2_tokens = norm_s2.split()

    alignments = _gen_alignments(norm_s1_tokens, norm_s2_tokens, s1_tokens,
                                 s2_tokens)

    return list(alignments)
