from scripts.annotation.an_annotate_error_type import annotate
from scripts.alignment.al_align_input_system import align_ref_system_basic
import os

# def annote_ref_sys(ref_path, sys_path, show_paths=False):
def annote_ref_sys(alignment_path, show_paths=False):

    # if not os.path.isdir('output'):
    #     os.mkdir('output')

    # Aligned files
    # alignment = read_alignment(alignment_path)
    # out_align_sys_ref = "output/align_sys_ref.tsv"
    # postprocess_all_alignment(alignment, out_align_sys_ref)

    # Annotated files
    # out_annot_sys_ref = "output/annot_sys_ref.tsv"

    # Alignment process
    # align_ref_system_basic(sys_path, ref_path, out_align_sys_ref)

    # Annotation process
    # annotation_lines = annotate(out_align_sys_ref, out_annot_sys_ref,
    #                             show_paths=show_paths)

    annotation_lines = annotate(alignment_path, f'{alignment_path}.align_sys_ref.tsv',
                                show_paths=show_paths)

    return annotation_lines


# TODO: we should do all the alignment in one shot.
# Now we will provide the alignment Bashar created and postprocess it 
# to reduce inserts and deletes to appends/prepends

# def read_alignment(path):
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

# def postprocess_alignment(src, tgt):
#     assert len(src) == len(tgt)

#     i, j = 0, 0
#     new_src, new_tgt = [], []
#     tags = []

#     prepend_src, prepend_tgt = [], []

#     while i < len(src) and j < len(tgt):
#         if src[i] == tgt[j]: # Keep

#             if prepend_tgt: 
#                 new_tgt.append(" ".join(prepend_tgt) + ' ' + tgt[j])
#             else:
#                 new_tgt.append(tgt[j])

#             if prepend_src:
#                 new_src.append(" ".join(prepend_src) + ' ' +src[i])

#             else:
#                 new_src.append(src[i])

#             i += 1
#             j += 1
#             prepend_src, prepend_tgt = [], []

#         else:
#             if src[i] != '' and tgt[j] != '': # Replace

#                 if prepend_tgt: 
#                     new_tgt.append(" ".join(prepend_tgt) + ' ' + tgt[j])
#                 else:
#                     new_tgt.append(tgt[j])

#                 if prepend_src:
#                     new_src.append(" ".join(prepend_src) + ' ' +src[i])

#                 else:
#                     new_src.append(src[i])

#                 i += 1
#                 j += 1
#                 prepend_src, prepend_tgt = [], []
#             else:
#                 prepend_src, prepend_tgt = [], []

#                 while i < len(src) and j < len(tgt) and src[i] == '' and tgt[j] != '':

#                     prepend_tgt.append(tgt[j])
#                     j += 1
#                     i += 1

#                 while i < len(src) and j < len(tgt) and src[i] != '' and tgt[j] == '':
#                     prepend_src.append(src[i])
#                     i += 1
#                     j += 1

#     if prepend_tgt:
#         new_tgt[-1] = new_tgt[-1] + ' ' + ' '.join(prepend_tgt)

#     if prepend_src:
#         new_src[-1] = new_src[-1] + ' ' + ' '.join(prepend_src)


#     assert len(new_tgt) == len(new_src)
#     assert " ".join(new_tgt).split() == " ".join(tgt).split()
#     assert " ".join(new_src).split() == " ".join(src).split()

#     return new_src, new_tgt

# def postprocess_alignment_no_span(src, tgt):
#     assert len(src) == len(tgt)

#     i, j = 0, 0
#     new_src, new_tgt = [], []
#     tags = []

#     prepend_tgt = []

#     while i < len(src) and j < len(tgt):
#         if src[i] == tgt[j]: # Keep

#             if prepend_tgt: 
#                 new_tgt.append(" ".join(prepend_tgt) + ' ' + tgt[j])
#             else:
#                 new_tgt.append(tgt[j])

#             new_src.append(src[i])

#             i += 1
#             j += 1
#             prepend_tgt = []

#         elif src[i] != '' and tgt[j] != '': # Replace

#             if prepend_tgt: 
#                 new_tgt.append(" ".join(prepend_tgt) + ' ' + tgt[j])
#             else:
#                 new_tgt.append(tgt[j])

#             new_src.append(src[i])

#             i += 1
#             j += 1
#             prepend_tgt = []

#         elif src[i] == '' and tgt[j] != '': # inserts
#             prepend_tgt = []

#             while i < len(src) and j < len(tgt) and src[i] == '' and tgt[j] != '':

#                 prepend_tgt.append(tgt[j])
#                 j += 1
#                 i += 1

#         else: # Deletions
#             new_src.append(src[i])
#             new_tgt.append(tgt[i])
#             j += 1
#             i += 1


#     if prepend_tgt:
#         new_tgt[-1] = new_tgt[-1] + ' ' + ' '.join(prepend_tgt)


#     # assert len(new_tgt) == len(new_src)
#     assert " ".join(new_tgt).split() == " ".join(tgt).split()
#     assert " ".join(new_src).split() == " ".join(src).split()

#     return new_src, new_tgt

# def enrich_alignment(src, tgt, tags):
#     assert len(src) == len(tgt) == len(tags)

#     i, j = 0, 0
#     new_src, new_tgt, new_tags = [], [], []

#     append_tgt = []
#     append_tag = []

#     # add <bos> and </eos> tokens to the beginning of 
#     # src and target
#     src = ['<bos>'] + src + ['</eos>']
#     tgt = ['<bos>'] + tgt + ['</eos>']
#     tags = ['UC'] + tags + ['UC']

#     while i < len(src) and j < len(tgt):
#         if src[i] == tgt[j]: # Keep

#             if append_tgt: # In case we caught an insert, append to current token
#                 new_tgt[-1]  = new_tgt[-1] + ' ' + ' '.join(append_tgt)

#                 if new_tags[-1] != 'UC': # update the tag 
#                     new_tags[-1]  = new_tags[-1] + '+' + '+'.join(append_tag)
#                 else:
#                     new_tags[-1] = '+'.join(append_tag)

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

#                 else:
#                     new_tags[-1] = '+'.join(append_tag)

#                 append_tgt = []
#                 append_tag = []

#             new_tgt.append(tgt[j])
        
#             tag = '+'.join([f'REPLACE_{t}' for t in tags[i].split('+')])
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
#         new_tags[-1] = new_tags[-1] + ' ' + ' '.append(append_tag)

#     assert len(new_tgt) == len(new_src)
#     assert " ".join(new_tgt).split() == " ".join(tgt).split()
#     assert " ".join(new_src).split() == " ".join(src).split()
#     assert len(new_src) == len(new_tags)

#     return new_src, new_tgt, new_tags


# def postprocess_all_alignment(alignment, path):
#     with open(path, mode='w') as f:
#         f.write(f'SOURCE\tTARGET\n')
#         for example in alignment:
#             src, tgt = [x[0] for x in example],  [x[1] for x in example]
#             tags =  [x[2] for x in example]
#             src_, tgt_, tags_ = postprocess_alignment(src, tgt)
#             # src_, tgt_ = postprocess_alignment_no_span(src, tgt)

#             for src_token, tgt_token, tag in zip(src_, tgt_, tags_):
#                 f.write(f'{src_token}\t{tgt_token}\t{tag}')
#                 f.write('\n')
#             f.write('\n')
