from sqlite3 import paramstyle
import sys
import codecs
from getopt import getopt
import argparse
from scripts.annotation.an_annote_sys_ref import annote_ref_sys
from camel_tools.utils.charsets import UNICODE_PUNCT_SYMBOL_CHARSET
import string


PUNCS = list(string.punctuation) + list(UNICODE_PUNCT_SYMBOL_CHARSET)

def collate_alignments(alignments):
    example = []
    examples = []

    for line in alignments:
        line = line.replace('\n', '').split('\t')
        if len(line) > 1:
            s, t, tag = line[0], line[1], line[2]
            example.append((s, t, tag))
        else:
            examples.append(example)
            example = []

    if example:
        examples.append(example)

    return examples


# def shrink_rich_tag(tag):
#     """
#     TODO: fix this! the shrinking here doesn't handle shrinking ^2 and ^* cases!!
#     But this doesn't happen in the data
#     """
#     if 'UNK' in tag:
#         return 'UNK'

#     from itertools import groupby
#     tag_ = sorted(tag.split('+'))
#     grouped_ops = [(k, len(list(g))) for k, g in groupby(tag_)]


#     assert '+'.join([x for el in grouped_ops for x in [el[0] for _ in range(el[1])]]) == '+'.join(sorted(tag.split('+')))
#     shrunk_tag = []
#     for k, rep in grouped_ops:
#         if rep == 1:
#             shrunk_tag.append(k)
#         elif rep == 2:
#             shrunk_tag.append(f'{k}^2')
#         elif rep > 2:
#             shrunk_tag.append(f'{k}^*')

#     return '+'.join(shrunk_tag)


def enrich_alignment(src, tgt, tags):
    assert len(src) == len(tgt) == len(tags)

    i, j = 0, 0
    new_src, new_tgt, new_tags = [], [], []

    append_tgt = []
    append_tag = []


    while i < len(src) and j < len(tgt):
        if src[i] == tgt[j]: # Keep

            new_tgt.append(tgt[j])
            new_tags.append(tags[i])
            new_src.append(src[i])

            i += 1
            j += 1

        elif src[i] != '' and tgt[j] != '': # Replace
            new_tgt.append(tgt[j])
            tag = '+'.join([f'REPLACE_{t}' for t in tags[i].split('+')])
            new_tags.append(tag)
            new_src.append(src[i])

            i += 1
            j += 1

        elif src[i] == '' and tgt[j] != '': # Insert
            tag = '+'.join([f'INSERT_{t}' for t in tags[i].split('+')])

            new_tgt.append(tgt[j])
            new_src.append(src[i])
            new_tags.append(tag)

            j += 1
            i += 1

        else: # Deletions
            new_src.append(src[i])
            new_tgt.append(tgt[i])

            if src[i] in PUNCS:
                new_tags.append('DELETE_PM')
            else:
                new_tags.append('DELETE_XM')

            j += 1
            i += 1


    assert len(new_tgt) == len(new_src)
    assert " ".join(new_tgt).split() == " ".join(tgt).split()
    assert " ".join(new_src).split() == " ".join(src).split()
    assert len(new_src) == len(new_tags)

    return new_src, new_tgt, new_tags

# def enrich_alignment(src, tgt, tags):
#     assert len(src) == len(tgt) == len(tags)

#     i, j = 0, 0
#     new_src, new_tgt, new_tags = [], [], []

#     append_tgt = []
#     append_tag = []

#     # add <bos> and </eos> tokens to the beginning of 
#     # src and target
#     src = ['<bos>'] + src + ['<eos>']
#     tgt = ['<bos>'] + tgt + ['<eos>']
#     tags = ['UC'] + tags + ['UC']

#     while i < len(src) and j < len(tgt):
#         if src[i] == tgt[j]: # Keep

#             if append_tgt: # In case we caught an insert, append to current token
#                 new_tgt[-1]  = new_tgt[-1] + ' ' + ' '.join(append_tgt)

#                 if new_tags[-1] != 'UC': # update the tag 
#                     new_tags[-1]  = new_tags[-1] + '+' + '+'.join(append_tag)
#                     # new_tags[-1] = shrink_rich_tag(new_tags[-1])
#                 else:
#                     new_tags[-1] = '+'.join(append_tag)
#                     # new_tags[-1] = shrink_rich_tag(new_tags[-1])

#                 append_tgt = []
#                 append_tag = []

#             new_tgt.append(tgt[j])
#             new_tags.append(tags[i])
#             new_src.append(src[i])

#             i += 1
#             j += 1

#         elif src[i] != '' and tgt[j] != '': # Replace

#             if append_tgt: # In case we caught an insert, append to current token
#                 new_tgt[-1]  = new_tgt[-1] + ' ' + ' '.join(append_tgt)

#                 if new_tags[-1] != 'UC': # update the tag
#                     tag = []
#                     for t in new_tags[-1].split('+'):
#                         if (not t.startswith('REPLACE') and not t.startswith('INSERT')
#                             and not t.startswith('DELETE')):
#                             tag.append(f'REPLACE_{t}')
#                         else:
#                             tag.append(t)

#                     tag = '+'.join(tag)
#                     new_tags[-1]  = tag + '+' + '+'.join(append_tag)
#                     # new_tags[-1] = shrink_rich_tag(new_tags[-1])

#                 else:
#                     new_tags[-1] = '+'.join(append_tag)
#                     # new_tags[-1] = shrink_rich_tag(new_tags[-1])

#                 append_tgt = []
#                 append_tag = []

#             new_tgt.append(tgt[j])

#             tag = '+'.join([f'REPLACE_{t}' for t in tags[i].split('+')])
#             # new_tags.append(shrink_rich_tag(tag))
#             new_tags.append(tag)

#             new_src.append(src[i])

#             i += 1
#             j += 1

#         elif src[i] == '' and tgt[j] != '': # Track all the inserts
#             append_tgt = []
#             append_tag = []

#             while i < len(src) and j < len(tgt) and src[i] == '' and tgt[j] != '':

#                 append_tgt.append(tgt[j])
#                 append_tag.append(f'INSERT_{tags[i]}')

#                 j += 1
#                 i += 1

#         else: # Deletions
#             new_src.append(src[i])
#             new_tgt.append(tgt[i])

#             new_tags.append('DELETE')

#             j += 1
#             i += 1

#     if append_tgt:
#         new_tgt[-1] = new_tgt[-1] + ' ' + ' '.join(append_tgt)
#         new_tags[-1] = new_tags[-1] + ' ' + ' '.join(append_tag)
#         # new_tags[-1] = shrink_rich_tag(new_tags[-1])

#     assert len(new_tgt) == len(new_src)
#     assert " ".join(new_tgt).split() == " ".join(tgt).split()
#     assert " ".join(new_src).split() == " ".join(src).split()
#     assert len(new_src) == len(new_tags)

#     return new_src, new_tgt, new_tags


def postprocess_all_alignment(alignment, path):
    with open(path, mode='w') as f:
        f.write(f'SOURCE\tTARGET\n')
        for example in alignment:
            src, tgt = [x[0] for x in example],  [x[1] for x in example]
            tags =  [x[2] for x in example]
            src_, tgt_, tags_ = enrich_alignment(src, tgt, tags)

            tags_ = [t.replace('REPLACE_SP', 'MERGE').replace('REPLACE_MG', 'SPLIT')
                     for t in tags_]

            unk_idx = [i for i, t in enumerate(tags_) if 'UNK' in t]

            for idx in unk_idx:
                tags_[idx] = 'UNK'

            for src_token, tgt_token, tag in zip(src_, tgt_, tags_):
                # tag = shrink_rich_tag(tag)
                tag = '+'.join(sorted(tag.split('+')))
                f.write(f'{src_token}\t{tgt_token}\t{tag}')
                f.write('\n')
            f.write('\n')


def print_usage():
    print("Usage: annotate_err_type_ar.py [OPTIONS] source target")
    print("where")
    print("  reference          -   the reference file")
    print("  target          -   the system's output")
    print("OPTIONS")

    print(
        "        --output  	                  -  The output file. "
        "Otherwise, it prints to standard output ")

parser = argparse.ArgumentParser()
# parser.add_argument('--sys_path', type=str, required=True,
#                     help="System's Output")

# parser.add_argument('--ref_path', type=str, required=True,
#                     help="Reference file")
parser.add_argument('--alignment', type=str, required=True,
                    help="Word-level alignment file.")

parser.add_argument('--output_path', type=str,
                    help="Output file path")

parser.add_argument('--enriched_output_path', type=str)

parser.add_argument('--show_edit_paths', action="store_true",
                    help="Whether to show the orthographic and "
                    "morphological edits paths")

args = parser.parse_args()

# lines = annote_ref_sys(args.ref_path, args.sys_path, args.show_edit_paths)
lines = annote_ref_sys(args.alignment, args.show_edit_paths)
alignments = collate_alignments(lines.split('\n')[:-2])
postprocess_all_alignment(alignments, args.enriched_output_path)

if args.output_path:
    write_output = codecs.open(args.output_path, 'w', "utf8")
    write_output.write(lines)
    write_output.close()
else:
    sys.stdout.write(lines)
