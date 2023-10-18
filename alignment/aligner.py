import re
import copy
import json
import editdistance
from ced_alignment import align_words
from utils import norm_alef_ya_teh
import argparse
import string
from camel_tools.utils.charsets import UNICODE_PUNCT_SYMBOL_CHARSET


PUNCS = list(string.punctuation) + list(UNICODE_PUNCT_SYMBOL_CHARSET)

def read_data(path):
    with open(path) as f:
        return [x.strip() for x in f.readlines()]


def read_alignment(path):
    example = []
    examples = []
    with open(path, mode='r') as f:
        for line in f.readlines():
            line = line.strip()
            if line:
                data = line.split('\t')
                ex = data[:-1] + [eval(data[-1])]
                example.append(ex)
            else:
                examples.append(example)
                example = []

        # adding the last example
        if example:
            examples.append(example)
    return examples


class BuggyRange:
    def __init__(self, start, end, ops):
        self.start = start
        self.end = end
        self.ops = ops

    def __repr__(self):
        return str(self.to_dict())

    def to_json_str(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output


def capture_bug(alignment):
    """Given a list of word-level alignments, capture the sequences of alignments
    that includes inserts, deletes, and replaces.

    Args
        alignment (list of tuples): list of tuples where each tuple represents a
            word-level alignment.

    Returns:
        A list of BuggyRange objects.
    """

    i = 0
    buggy_span = []

    while i < len(alignment):
        potential_buggy = {}
        start_idx = i - 1

        # if we see a sequence of deletes, replaces, and inserts,
        # this is a potential bug
        while i < len(alignment) and ('DELETE' in alignment[i][3] or
                                      'REPLACE' in alignment[i][4] or
                                      'INSERT' in alignment[i][3]):

            potential_buggy[i] = alignment[i]
            i += 1

        if len(potential_buggy) > 1:
            # save the start and end anchors and the sequence of edits
            buggy_span.append(BuggyRange(start_idx, i, potential_buggy))

        elif len(potential_buggy) == 0:
            i += 1

    return buggy_span


def construct_src_tgt(buggy_range):
    """Given a sequence of buggy alignments, construct the source and the
    target.

    Args:
        buggy_range (BuggyRange): an instance of a buggy range objects

    Returns:
        A tuple of lists containing the src and tgt tokens respectively.
    """

    src = []
    tgt = []
    for v in buggy_range.ops.values():

        if v[3] == 'DELETE':
            src.append(v[-2])
            tgt.append('NIL')

        elif v[3] == 'INSERT':
            src.append('NIL')
            tgt.append(v[-1])

        elif v[4] == 'REPLACE':
            src.append(v[2] if v[-2] == '' else v[-2])
            tgt.append(v[3] if v[-1] == '' else v[-1])

    return src, tgt


def consruct_clean_src_tgt(align):
    """Given a sequence of clean (i.e., not buggy) alignments, construct the
    source and target.

    Args:
        align (list of tuple): list of tuples where each tuple represent a
            word-level alignment.

    Returns:
        A tuple of lists containing the src and tgt tokens respectively.
    """

    src_tokens = []
    tgt_tokens = []

    for x in align:
        if x[4] == 'KEEP' or x[4] == 'REPLACE':
            # recovering the original tokens in case of normalization
            src_tokens.append(x[2] if x[-2] == '' else x[-2])
            tgt_tokens.append(x[3] if x[-1] == '' else x[-1])

        elif x[3] == 'INSERT':
            src_tokens.append('')
            tgt_tokens.append(x[-1])

        elif x[3] == 'DELETE':
            src_tokens.append(x[-2])
            tgt_tokens.append('')

    return src_tokens, tgt_tokens


def perfect_align(src, tgt):
    """Given src and tgt tokens, we try to find the optimal alignment
    by greedily resolving adjacent merges and splits.

    Args:
        src (list of str): src tokens.
        tgt (list of str): tgt tokens.

    Returns:
        A tuple of lists where the lists contains the optimal aligned
        src and tgt tokens, respectively.
    """

    assert len(src) == len(tgt)

    basic_edit = get_edit(src, tgt) # get the basic edit between src and tgt
    best_edit = {'src': src, 'tgt': tgt, 'edit': basic_edit}

    i = 0

    while i < len(src):

        if (src[i] == 'NIL' and tgt[i] != 'NIL'): # insert or a potential split

            prepend = is_split_merge(src, tgt, i, 'prepend', src_first=True)
            append = is_split_merge(src, tgt, i, 'append', src_first=True)
            best_edit = get_best_edit(best_edit, prepend, append)

        elif (src[i] != 'NIL' and tgt[i] == 'NIL'): # delete or a potential merge

            prepend = is_split_merge(tgt, src, i, 'prepend', src_first=False)
            append = is_split_merge(tgt, src, i, 'append', src_first=False)
            best_edit = get_best_edit(best_edit, prepend, append)

        # rewind the index in case of a change
        if best_edit['src'] != src or best_edit['tgt'] != tgt:
            src, tgt = best_edit['src'], best_edit['tgt']
            i = 0
            continue

        i += 1

    return best_edit['src'], best_edit['tgt']


def is_split_merge(src, tgt, i, mode='prepend', src_first=False):
    """Given src and tgt tokens, we construct new sequences by resolving
    adjancent merges and splits. In case of a feasbile merge/split, we compute
    the edit distance between the newly constructed src and tgt sequences.

    If the mode is `prepend`, we construct a new tgt sequence such that the
    latest non-NIL tgt token is prepended to tgt[i].
    Otherwise, if the mode is `prepend`, we construct a new tgt sequence such
    that the first non-NIL tgt token is appended to tgt[i].


    Args:
        src (list of str): src tokens.
        tgt (list of str): tgt tokens.
        i (int): the index where src[i] == NIL.
        mode (str): a flag to control the construction of the new tgt sequence.
        src_first (bool): a flag to indicate which newly sequence should be
            returned first. If `True`, we return src first. Otherwise, we
            return tgt first. This is needed so we can use the same function
            to handle both splits and merges.

    Returns:
        A dict containing the newly constructed src and tgt sequences along
        with their edit distance, where src and tgt are represented as list
        of str.
    """

    if mode == 'prepend':
        j = i - 1

        while j >= 0 and tgt[j] == 'NIL': # find the latest non-NIL 
            j -= 1

        if j >= 0:
            src_ = src[:i] + src[i + 1:]
            tgt_ = (tgt[:j] + [tgt[j] + ' ' + tgt[i]] +
                    [tgt[x] for x in range(j+1, len((tgt))) if x != i])

            if len(src_) == len(tgt_):
                edit_all = get_edit(src_, tgt_)
                edit_no_space = get_edit(src_, [x.replace(' ','') for x in tgt_])

                # give a small weight to the added spaces
                edit = edit_no_space + 0.1 * (edit_all - edit_no_space)

                if src_first:
                    # split on target
                    return {'src': src_, 'tgt': tgt_,
                            'edit': edit}
                else:
                    # merge on source
                    return {'src': tgt_ , 'tgt': src_,
                            'edit': edit}

        if src_first:
            return {'src': src, 'tgt': tgt, 'edit': get_edit(src, tgt)}
        else:
            return {'src': tgt, 'tgt': src, 'edit': get_edit(src, tgt)}


    elif mode == 'append': # find the first non-NIL
        j = i + 1

        while j < len(tgt) and tgt[j] == 'NIL':
            j += 1

        if j < len(tgt):
            src_ = src[:i] + src[i + 1:]

            tgt_ = (tgt[:i] + [tgt[i] + ' ' + tgt[j]] +
                    [tgt[x] for x in range(i+1, len((tgt))) if x != j])

            if len(src_) == len(tgt_):
                edit_all = get_edit(src_, tgt_)
                edit_no_space = get_edit(src_, [x.replace(' ','') for x in tgt_])

                # give a small weight to the added spaces
                edit = edit_no_space + 0.1 * (edit_all - edit_no_space)

                if src_first:
                    # split on target
                    return {'src': src_, 'tgt': tgt_,
                            'edit': edit}
                else:
                    # merge on source
                    return {'src': tgt_ , 'tgt': src_,
                            'edit': edit}

        if src_first:
            return {'src': src, 'tgt': tgt, 'edit': get_edit(src, tgt)}
        else:
            return {'src': tgt, 'tgt': src, 'edit': get_edit(src, tgt)}


def get_best_edit(edit1, edit2, edit3):
    """Compares three edit distances together and returns the minimum.
    In case of a tie, always prefer the first edit"""

    edits = (edit1, edit2, edit3)
    # in case of a tie, prefer the basic edit
    if edit1['edit'] == edit2['edit'] == edit3['edit']:
        return edit1

    edits_w_idx = [(i, x['edit']) for i, x in enumerate(edits)]

    min_edit = min(edits_w_idx, key=lambda x: x[1])[0]

    return edits[min_edit]


def get_edit(src, tgt):
    """Computes the edit distance betweet src and tgt in a normalized space

    Args:
        src (list of str): src tokens.
        tgt (list of str): tgt tokens.

    Returns:
        The cumulative edit distance between each pair of words in src and tgt.
    """

    edit = 0

    for i in range(len(src)):
        s, t = norm_alef_ya_teh(src[i]), norm_alef_ya_teh(tgt[i])
        edit += edits(s.replace('PNX','') if s != 'NIL' else '',
                      t.replace('PNX','') if t != 'NIL' else '')

    return edit


def edits(s1, s2):
    return editdistance.distance(s1, s2)


def bug_fix(align, seq_bug):
    """Given an alignment and the list of buggy sequences in it,
    generate aligned source and target sequences.

    Args:
        align (list of tuples): list of tuples where each tuple represents a
            word-level alignment.
        seq_bug (list of BuggyRange): list of BuggyRange objects.

    Returns:
        A tuple of lists containing the src and tgt tokens respectively.
    """

    src = []
    tgt = []
    start = 0

    for bug in seq_bug:
        # everything before the potential bug is clean sequence of alignment
        src_tokens, tgt_tokens = consruct_clean_src_tgt(align[start: bug.start + 1])
        src += src_tokens
        tgt += tgt_tokens

        # construct the source and target for the buggy sequence
        p_src, p_tgt = construct_src_tgt(bug)
        # fix their alignment
        _p_src, _p_tgt = perfect_align(p_src, p_tgt)

        src += _p_src
        tgt +=  _p_tgt

        start = bug.end

    src_tokens, tgt_tokens = consruct_clean_src_tgt(align[start: ])
    src += src_tokens
    tgt += tgt_tokens

    assert len(src) == len(tgt)
    return src, tgt


def post_process_alignment(alignment):
    """Processes word-level alignment which were computed using
    a weighted character-level Levenshtein edit distance to generate
    many-to-one, one-to-many, and many-to-many alignments.

    Args:
        alignment:  a list of tuples where each tuple `t`
            contains the followings:
            t[0] the src idx or None in case of insertion.
            t[1] the tgt idx or None in case of deletion.
            t[2] the src token in normalized space for pnx and nums.
            t[3] the tgt token in normalized space for pnx and nums.
            t[4] the edit operation.
            t[5] the original src token.
            t[6] the original tgt token.
            Note: In case of deletion or insertion, the length of each tuple
            will be 6 instead of 7 since we won't have src or tgt tokens.
    """

    clean_alignment = []
    for i, align in enumerate(alignment):
        seq_bug = capture_bug(align)

        if len(seq_bug) == 0:
            src, tgt = consruct_clean_src_tgt(align)
            clean_alignment.append({'src': src, 'tgt' : tgt})

        else:
            src, tgt = bug_fix(align, seq_bug)
            clean_alignment.append({'src': src, 'tgt': tgt})

    # a final pass on the alignment to reduce inserts followed by 
    # deletions to replaces
    improved_alignment = reduce_inserts_deletions(clean_alignment)

    return improved_alignment


def reduce_inserts_deletions(alignment):
    """Given a clean alignment, we will reduce the sequences
    of inserts followed by deletions to replaces.

    Args:
        alignment (list of dict): a list of dicts where each dict contains
            the aligned src and tgt tokens.

    Returns:
        A list of dict containing the reduced alignments.
    """

    reduced_alignment = []

    for example in alignment:
        assert len(example['src']) == len(example['tgt'])
        src, tgt = example['src'], example['tgt']

        src = [x.replace('PNX', '').replace('NIL','') for x in src]
        tgt = [x.replace('PNX', '').replace('NIL','') for x in tgt]

        i = 0
        s_idx = 0
        d_idx = 0
        new_align = []

        while i < len(src):
            src_token = src[i]
            tgt_token = tgt[i]

            if src_token == '' and tgt_token != '': # insertion
                s_idx = i

                # get all insertions
                while s_idx < len(src) and src[s_idx] == '' and tgt[s_idx] != '':
                    s_idx += 1

                d_idx = s_idx

                # get all deletions
                while d_idx < len(src) and src[d_idx] != '' and tgt[d_idx] == '':
                    d_idx += 1

                if d_idx != s_idx:
                    span = list(zip(src[i:d_idx], tgt[i:d_idx]))
                    reduced_span = reduce_span(span)
                    new_align += reduced_span
                    i = d_idx

                else:
                    new_align.append((src[i], tgt[i]))
                    i += 1
            else:
                new_align.append((src[i], tgt[i]))
                i += 1

        reduced_alignment.append({'src': [x[0] for x in new_align],
                                  'tgt': [x[1] for x in new_align]
                                 })

    return reduced_alignment

def reduce_span(span):
    """Given a span of aligned src and tgt tokens representing inserts
    and deletions, attempt to combine them in a replace monotonically.

    A replace is valid if one of the following applies:
        1) tgt is a pnx and src is a single char
        2) src is a pnx and src is a single char
        3) tgt is a pnx and src is a pnx
        4) tgt is a word and src is a word

    Args:
        span (list of tuples): a list of tuples where each tuple contains
            aligned src and tgt tokens.

    Returns:
        A list of tuples that represent replacements if a reduction is feasible.
    """

    inserts = [x for i, x in enumerate(span) if (x[0] == '' and x[1] != '')]
    deletes = [x for i, x in enumerate(span) if (x[0] != '' and x[1] == '')]

    i_idx, d_idx = 0, 0
    replaces = []

    while i_idx < len(inserts) and d_idx < len(deletes):
        insert = inserts[i_idx]
        delete =  deletes[d_idx]

        tgt = insert[1]
        src = delete[0]

        if tgt in PUNCS and len(src) == 1:
            replaces.append((src, tgt))
            i_idx += 1
            d_idx += 1

        elif src in PUNCS and len(tgt) == 1:
            replaces.append((src, tgt))
            i_idx += 1
            d_idx += 1

        elif tgt not in PUNCS and src not in PUNCS:
            replaces.append((src, tgt))
            i_idx += 1
            d_idx += 1

        elif src in PUNCS and tgt in PUNCS:
            replaces.append((src, tgt))
            i_idx += 1
            d_idx += 1

        else:
            if len(deletes) > len(inserts):
                replaces.append(delete)
                d_idx += 1
            else:
                replaces.append(insert)
                i_idx += 1

    while i_idx < len(inserts):
        replaces.append(inserts[i_idx])
        i_idx += 1

    while d_idx < len(deletes):
        replaces.append(deletes[d_idx])
        d_idx += 1

    if replaces:
        return replaces

    return None


def write_data(alignment, path):
    with open(path, mode='w') as f:
        f.write('SOURCE\tTARGET')
        f.write('\n')
        for example in alignment:
            for s, t in zip(example['src'], example['tgt']):
                s = s.replace('PNX', '').replace('NIL', '')
                t = t.replace('PNX', '').replace('NIL', '')

                f.write(f'{s}\t{t}')
                f.write('\n')
            f.write('\n')


def verify(src_sents, tgt_sents, alignment):
    """Verifying that we are able to recover the src and tgt sents from the
    alignment.

    Args:
        src_sents (list of str): list of src sentences.
        tgt_sents (list of str): list of tgt sentences.
        alignment (list of dict): list of dicts containing the aligned src and
            tgt tokens.
    """

    assert len(src_sents) == len(tgt_sents) == len(alignment)
    for src, tgt, align in zip(src_sents, tgt_sents, alignment):
        src_ = ' '.join([x.replace('PNX', '').replace('NIL','') for x in align['src']])
        tgt_ = ' '.join([x.replace('PNX', '').replace('NIL','') for x in align['tgt']])

        # if re.sub(' +', ' ', src).strip() != re.sub(' +', ' ', src_).strip():
        #     import pdb; pdb.set_trace()

        # if re.sub(' +', ' ', tgt).strip() != re.sub(' +', ' ', tgt_).strip():
        #     import pdb; pdb.set_trace()
        assert re.sub(' +', ' ', src).strip() == re.sub(' +', ' ', src_).strip()
        assert re.sub(' +', ' ', tgt).strip() == re.sub(' +', ' ', tgt_).strip()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, help='Source file.', required=True)
    parser.add_argument('--tgt', type=str, help='Target file.', required=True)
    parser.add_argument('--output', type=str, help='Output file.', required=True)
    args = parser.parse_args()

    src_sents = read_data(args.src)
    tgt_sents = read_data(args.tgt)

    # create basic alignments first
    basic_alignment = []
    for i, (src, tgt) in enumerate(zip(src_sents, tgt_sents)):
        b_align = align_words(src, tgt)
        basic_alignment.append(b_align)

    # clean up the basic alignment
    clean_alignment = post_process_alignment(basic_alignment)
    verify(src_sents, tgt_sents, clean_alignment)
    write_data(clean_alignment, args.output)

