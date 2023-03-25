import argparse
import copy


def read_lines(path):
    with open(path) as f:
        return [x.strip() for x in f.readlines()]


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


def create_m2_edits_per_ex(example):
    """
    Takes an alignment for a sentence and generates its m2 edits.
    """
    edits = []
    src_idx = 0
    preds = []
    for i, (s, t) in enumerate(example):
        if s == t:
            src_idx += 1
            continue

        if s != '' and t != '':
            edit = f'A {src_idx} {src_idx + len(s.split())}|||Replace|||{t}|||REQUIRED|||-NONE-|||0'
            src_idx += len(s.split())
            preds.append(t)

        elif s == '' and t != '':
            edit = f'A {src_idx} {src_idx}|||Insert|||{t}|||REQUIRED|||-NONE-|||0'
            preds.append(t)

        elif s != '' and t == '':
            edit = f'A {src_idx} {src_idx + len(s.split())}|||Delete||||||REQUIRED|||-NONE-|||0'
            src_idx += len(s.split())
            preds.append('')

        edits.append(edit)

    return "\n".join(edits), preds


def create_m2_edits(examples):
    edits = []
    preds = []
    for i, example in enumerate(examples):
        ex_edits, ex_preds = create_m2_edits_per_ex(example)
        edits.append(ex_edits)
        preds.append(ex_preds)

    return edits, preds


def write_edits(src_sentences, edits, path):
    with open(path, mode='w') as f:
        for i, edit in enumerate(edits):
            f.write(f'S {src_sentences[i]}\n')
            if edit:
                f.write(edit)
                f.write('\n')
                f.write('\n')
            else:
                f.write('\n')


def post_process_edit(m2_edits, tgt_sent):
    """
    Takes the edits for the sentence and the original target and
    post-process the edits to deal with any anomaly that happened during
    normalization for the alignment
    """
    m2_edits = m2_edits.split('\n')
    new_m2_edits = []
    curr_idx = 0
    tgt_idx = 0
    tgt_words = tgt_sent.split()

    for i, edit in enumerate(m2_edits):
        edit = edit.split('|||')
        new_edit = copy.copy(edit)
        span = edit[0].split()[1:]
        start, end = int(span[0]), int(span[1])
        tgt_idx += start - curr_idx

        # recover the original target token that has to be inserted or replaced
        if edit[1] == 'Insert' or edit[1] == 'Replace':
            current_fix = edit[2].split()
            new_edit[2] = " ".join(tgt_words[tgt_idx: tgt_idx + len(current_fix)])
            tgt_idx += len(current_fix)

        curr_idx = end
        new_m2_edits.append('|||'.join(new_edit))

    return "\n".join(new_m2_edits)


def postprocess(edits, tgt_sents):
    fixed_edits = []
    for i, (edit, tgt_sent) in enumerate(zip(edits, tgt_sents)):
        if edit:
            fixed_edits.append(post_process_edit(edit, tgt_sent))
        else:
            fixed_edits.append('')
    return fixed_edits


def recover_tgt(src, m2_edits):
    """Given a src and a set of m2 edits, recover the target"""

    m2_edits = m2_edits.split('\n')
    tgt = []
    curr_idx = 0

    for edit in m2_edits:
        edit = edit.split('|||')
        span = edit[0].split()[1:]
        start, end = int(span[0]), int(span[1])
        tgt += src[curr_idx: start]

        if edit[1] != 'Delete':
            tgt.append(edit[2])

        curr_idx = end

    if curr_idx < len(src):
        tgt += src[curr_idx: ]

    return " ".join(tgt)

def recover_and_compare_tgts(m2_edits, src_sents, tgt_sents):
    """
    Given src sentences and m2 edits, it recovers the target sentences
    and verifies that we are able to get the targets
    """
    for i in range(len(m2_edits)):
        if m2_edits[i]:
            tgt = recover_tgt(src_sents[i].split(), m2_edits[i])
            assert tgt == tgt_sents[i]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, help='Source sentences.')
    parser.add_argument('--tgt', type=str, help='Target sentences.')
    parser.add_argument('--align', type=str, help='Alignment file.')
    parser.add_argument('--output', type=str, help='Output file.')
    args = parser.parse_args()

    src_sentences = read_lines(args.src)
    tgt_sentences = read_lines(args.tgt)

    alignment = read_alignment(args.align)
    edits, _ = create_m2_edits(alignment)
    clean_edits = postprocess(edits, tgt_sentences)

    recover_and_compare_tgts(clean_edits, src_sentences, tgt_sentences)
    write_edits(src_sentences, edits, args.output)
