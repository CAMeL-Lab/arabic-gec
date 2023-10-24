from nltk.metrics import edit_distance_align
import re
import argparse


def read_lines(path):
    with open(path, mode='r', encoding='utf8') as f:
        return [x for x in f.readlines()]

def levenshtein_alignment(src_sents, tgt_sents):
    assert len(src_sents) == len(tgt_sents)
    alignment = []

    for src, tgt in zip(src_sents, tgt_sents):
        path = edit_distance_align(src, tgt, substitution_cost=2)
        ex_alignment = vanilla_alignment(path, src, tgt)
        alignment.append(ex_alignment)

    return alignment


def vanilla_alignment(path: list, source: str, target: str,
          empty_token: str="@") -> None:
    """Prints the alignment for the path

    Args:
     path: list of (row, column) tuples
     source: the source string
     target: the target string
     empty_token: token to insert for skipped characters
    """
    previous_row = previous_column = None
    source_tokens = []
    target_tokens = []
    source = empty_token + source
    target = empty_token + target

    for current_row, current_column in path[1:]:
        source_token = source[current_row] if current_row != previous_row else empty_token
        target_token = target[current_column] if current_column != previous_column else empty_token

        source_tokens.append(source_token)
        target_tokens.append(target_token)

        previous_row, previous_column = current_row, current_column

    # collapsing the characters into word alignments
    alignment = [[]]
    for src_token, tgt_token in zip(source_tokens, target_tokens):
        if src_token == tgt_token == ' ':
            src_word = [x[0] for x in alignment[-1] if x[0] != empty_token]
            tgt_word = [x[1] for x in alignment[-1] if x[1] != empty_token]
            alignment[-1] = (''.join(src_word), ''.join(tgt_word))

            alignment.append([])
        else:
            alignment[-1].append((src_token, tgt_token))

    src_word = [x[0] for x in alignment[-1] if x[0] != empty_token]
    tgt_word = [x[1] for x in alignment[-1] if x[1] != empty_token]
    alignment[-1] = (''.join(src_word), ''.join(tgt_word))

    return alignment


def read_alignment(path):
    example = []
    examples = []
    with open(path) as f:
        for line in f.readlines()[1:]:
            line = line.replace('\n', '').split('\t')
            if len(line) > 1:
                s, t = line
                example.append((s, t))
            else:
                examples.append(example)
                example = []

        if example:
            examples.append(example)

    return examples


def paragraphs(lines, is_separator=lambda x : x == '\n', joiner=''.join):
    paragraph = []
    for line in lines:
        if is_separator(line):
            if paragraph:
                yield joiner(paragraph)
                paragraph = []
        else:
            paragraph.append(line)
    if paragraph:
        yield joiner(paragraph)


def m2_file_alignment(m2_file):
    alignment = []
    m2_lines = read_lines(m2_file)

    for example in paragraphs(m2_lines):
        sent_alignment = []
        idx = 0
        lines = example.strip().split('\n')
        assert lines[0].startswith('S ')
        src_tokens = lines[0][2:].split()

        for line in lines[1:]:
            assert line.startswith('A ')

            line = line[2:]
            fields = line.split('|||')
            start_offset = int(fields[0].split()[0])
            end_offset = int(fields[0].split()[1])
            etype = fields[1]

            if idx != start_offset: # adding correct tokens to the alignment 
                sent_alignment += list(zip(src_tokens[idx: start_offset], src_tokens[idx: start_offset]))


            corrections =  fields[2]
            original = ' '.join(src_tokens[start_offset:end_offset])

            sent_alignment.append((original, corrections))

            idx = end_offset

        if idx != len(src_tokens):
            sent_alignment += list(zip(src_tokens[idx: ], src_tokens[idx: ]))

        alignment.append(sent_alignment)

    return alignment


def verify(src_sents, tgt_sents, alignment):
    assert len(src_sents) == len(tgt_sents) == len(alignment)

    for src, tgt, align in zip(src_sents, tgt_sents, alignment):
        src_ = ' '.join([x[0] for x in align])
        tgt_ = ' '.join([x[1] for x in align])

        if re.sub(' +', ' ', src).strip() != re.sub(' +', ' ', src_).strip():
            import pdb; pdb.set_trace()

        if re.sub(' +', ' ', tgt).strip() != re.sub(' +', ' ', tgt_).strip():
            import pdb; pdb.set_trace()

        assert re.sub(' +', ' ', src).strip() == re.sub(' +', ' ', src_).strip()
        assert re.sub(' +', ' ', tgt).strip() == re.sub(' +', ' ', tgt_).strip()


def alignment_error_rate(reference, hypothesis, possible=None):
    """
    References:
        https://aclanthology.org/W03-0301.pdf
        https://aclanthology.org/J03-1002.pdf
        https://www.cis.uni-muenchen.de/~fraser/pubs/fraser_tr616_alignqual.pdf
        https://www.nltk.org/_modules/nltk/translate/metrics.html#alignment_error_rate
    """
    reference = frozenset(reference)
    hypothesis = frozenset(hypothesis)


    if possible is None:
        possible = reference
    else:
        assert reference.issubset(possible)  # sanity check

    return 1.0 - (len(hypothesis & reference) + len(hypothesis & possible)) / float(
        len(hypothesis) + len(reference)
    )

def precision_recall_f1(reference, hypothesis):
    reference = frozenset(reference)
    hypothesis = frozenset(hypothesis)

    p = len(hypothesis & reference) / len(hypothesis)
    r = len(hypothesis & reference) / len(reference)
    f1 = 2 * p * r / (p + r)

    return {'p': p, 'r': r, 'f1': f1}


def evaluate(m2_alignment, my_alignment):
    avg_p, avg_r, avg_f1, avg_aer = 0, 0, 0, 0

    assert len(m2_alignment) == len(my_alignment)

    for i in range(len(m2_alignment)):

        p_r_f1 = precision_recall_f1(m2_alignment[i], my_alignment[i])
        aer = alignment_error_rate(m2_alignment[i], my_alignment[i])

        avg_p += p_r_f1['p']
        avg_r += p_r_f1['r']
        avg_f1 += p_r_f1['f1']
        avg_aer += aer

    avg_p /= len(m2_alignment)
    avg_r /= len(m2_alignment)
    avg_f1 /= len(m2_alignment)
    avg_aer /= len(m2_alignment)

    print(f'Precision: {avg_p*100: .2f}')
    print(f'Recall:    {avg_r*100: .2f}')
    print(f'F1:        {avg_f1*100: .2f}')
    print(f'AER:       {avg_aer: .2f}')



def evaluate_all(m2_alignment, my_alignment, possible=None):

    assert len(m2_alignment) == len(my_alignment)
    
    if possible == None:
        possible = m2_alignment

    hyp_ref_intersection = 0
    hyp_possible_intersection = 0
    len_hyp = 0
    len_ref = 0
    
    for i in range(len(m2_alignment)):
        ref = frozenset(m2_alignment[i])
        hyp = frozenset(my_alignment[i])
        pos = frozenset(possible[i])
        
        hyp_ref_intersection += len(ref & hyp)
        hyp_possible_intersection += len(pos & hyp)
        len_hyp += len(hyp)
        len_ref += len(ref)

    aer = 1.0 - float(hyp_ref_intersection + hyp_possible_intersection) / float(len_hyp + len_ref)

    p = hyp_ref_intersection / len_hyp
    r = hyp_ref_intersection / len_ref
    f1 = 2 * p * r / (p + r)

    print(f'Precision: {p*100: .2f}')
    print(f'Recall:    {r*100: .2f}')
    print(f'F1:        {f1*100: .2f}')
    print(f'AER:       {aer: .2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', help='Source file')
    parser.add_argument('--tgt', help='Target file')
    parser.add_argument('--gold_alignment', help='Gold M2 file')
    parser.add_argument('--m2_alignment', help='Automatically generated m2 file')
    parser.add_argument('--areta_alignment', help='Areta generated alignment')
    parser.add_argument('--our_alignment', help='Our generated alignment')

    args = parser.parse_args()

    src_sents = read_lines(args.src)

    tgt_sents = read_lines(args.tgt)

    ref_alignment = m2_file_alignment(args.gold_alignment)

    m2_alignment = m2_file_alignment(args.m2_alignment)

    areta_alignment = read_alignment(args.areta_alignment)

    my_alignment = read_alignment(args.our_alignment)

    levenshtein = levenshtein_alignment(src_sents, tgt_sents)

    assert len(ref_alignment) == len(my_alignment) == len(levenshtein) == len(m2_alignment) == len(areta_alignment)

    verify(src_sents, tgt_sents, ref_alignment)
    verify(src_sents, tgt_sents, m2_alignment)
    verify(src_sents, tgt_sents, areta_alignment)
    verify(src_sents, tgt_sents, my_alignment)
    verify(src_sents, tgt_sents, levenshtein)


    print('Our Alignment:')
    evaluate_all(ref_alignment, my_alignment)
    print()
    print('Standard Levenshtein:')
    evaluate_all(ref_alignment, levenshtein)
    print()
    print('M2 Alignment:')
    evaluate_all(ref_alignment, m2_alignment)
    print()
    print('ARETA Alignment:')
    evaluate_all(ref_alignment, areta_alignment)