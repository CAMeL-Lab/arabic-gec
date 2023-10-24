# MIT License
#
# Copyright 2020-2021 New York University Abu Dhabi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""Produce word alignment within aligned sentences.

Usage:
    align_text.py (-s SOURCE | --source=SOURCE)
               (-t TARGET | --target=TARGET)
               (-m MODE | --mode=MODE)
               (-o OUTPUT | --out OUTPUT)
    align_text.py (-h | --help) 

Options:
  -s SOURCE --source=SOURCE
        source/reference/gold sentences file
  -t TARGET --target=TARGET
        target/hypothesis/prediction sentences file
  -m MODE --mode=MODE
        Two modes to choose from: 
            1- 'align' To produce full alignments (one-to-many and many-to-one)
            2- 'basic' To produce basic alignments with operation and distance details (one-to-one)
  -o OUTPUT --output=OUTPUT
        Prefix for single output files
  -h --help
        Show this screen.
"""


import rapidfuzz.distance.Levenshtein as editdistance
from docopt import docopt
from alignment import align_words


def write_exact_alignment_only(alignments, src_sent, trg_sent, src_stream, trg_stream, col_align_stream):

    keys = list(range(max(len(src_sent), len(trg_sent))))
    words = dict.fromkeys(keys)

    # initiate the words dictionary
    for key in words:
        words[key] = {}
        words[key]['src'] = []
        words[key]['trg'] = []


    i = 0

    # store the location of the last substitute alignment
    last_s_idx = -1
    last_non_d_op = ''
    last_non_d_idx = -1
    # go through all the alignments
    while i < len(alignments): 
        current_op = alignments[i][2]
        next_idx = i+1
        if next_idx >= len(alignments):
            next_op = ''
        else:
            next_op = alignments[next_idx][2]


        current_idx = i
        # a substitute followed by a delete, this means that the source token
        #   is aligned to nothing on the target side 
        if current_op == 's' and next_op == 'd':
            last_s_idx = i
            last_non_d_op = 's'
            last_non_d_idx = i
            current_trg_word = trg_sent[alignments[i][1] - 1]
            current_src_word = src_sent[alignments[i][0] - 1]
            hyp_src_word = [current_src_word] # source word tokens hypothesis
            current_dist = editdistance.distance(current_trg_word, current_src_word)
            continue_merge = False

            # keep concatenating "deleted" tokens until we reach a new operation
            #   or the end of the sentence or a change in the editdistance

            while next_op == 'd' and i < len(alignments) and continue_merge:
                i += 1
                temp_word = ''.join(hyp_src_word) + src_sent[alignments[i][0] - 1]
                new_dist = editdistance.distance(current_trg_word, temp_word)
                if new_dist >= current_dist:
                    i -= 1
                    continue_merge = False
                elif new_dist == 0:
                    hyp_src_word.append(src_sent[alignments[i][0] - 1])
                    continue_merge = False
                else:
                    # if the new distance is less than the current, assign it 
                    #   to the current and keep going
                    continue_merge = True
                    current_dist = new_dist
                    hyp_src_word.append(src_sent[alignments[i][0] - 1])
                    if (i+1) < len(alignments):
                        next_op = alignments[i+1][2]
            
            # if we exited because the distance became larger, rewind the index
            if continue_merge:
                if temp_word.endswith(hyp_src_word[-1]):
                    hyp_src_word = hyp_src_word[:-1]
                    i = i - 1

            words[alignments[current_idx][0] - 1]['src'].extend(hyp_src_word)
            words[alignments[current_idx][0] - 1]['trg'].append(current_trg_word)
            i += 1
        
        # a delete
        elif current_op == 'd':
            if last_non_d_op == 's':
                # try attaching it to the previous 's' and check if the edit distance decreases
                prev_dist = editdistance.distance(''.join(words[alignments[last_s_idx][0] - 1]['trg']), ''.join(words[alignments[last_s_idx][0] - 1]['src']))
                
                hypo = words[alignments[last_s_idx][0] - 1]['src'] + [src_sent[alignments[i][0] - 1]]
                hyp_dist = editdistance.distance(''.join(words[alignments[last_s_idx][0] - 1]['trg']), ''.join(hypo))
                if hyp_dist <= prev_dist:
                    words[alignments[last_s_idx][0] - 1]['src'] = hypo
                    i += 1
                    continue
            
            # this is a d/i sequence, that is instead of a substitute it is 
            #   treated as a delete followed by an insert
            if next_op == 'i':
                words[alignments[i][0] - 1]['src'].append(src_sent[alignments[i][0]-1])
                words[alignments[i][0] - 1]['trg'].append(trg_sent[alignments[i+1][1]-1])
                last_non_d_op = 's'
                last_non_d_idx = i
                i += 2
            # otherwise attach it to the next 's'
            elif next_op == 's':
                i += 1
                last_s_idx = i
                last_non_d_op = 's'
                last_non_d_idx = i
                words[alignments[i][0] - 1]['src'].append(src_sent[alignments[i-1][0]-1])
                words[alignments[i][0] - 1]['src'].append(src_sent[alignments[i][0]-1])
                words[alignments[i][0] - 1]['trg'].append(trg_sent[alignments[i][1]-1])
                i += 1
            
            # or keep going if the next token is also deleted
            elif next_op == 'd':
                current_src_word = src_sent[alignments[i][0] - 1]
                hyp_src_word = [current_src_word]
                current_dist = editdistance.distance('', current_src_word)
                done = False
                while next_op == 'd' and i+1 < len(alignments) and not done:
                    i += 1
                    temp_word = ''.join(hyp_src_word) + src_sent[alignments[i][0] - 1]
                    hyp_src_word.append(src_sent[alignments[i][0] - 1])
                    if (i+1) < len(alignments):
                        next_op = alignments[i+1][2]
                
                if next_op == 's':
                    i += 1
                    last_s_idx = i
                    last_non_d_op = 's'
                    last_non_d_idx = i
                    hyp_src_word.append(src_sent[alignments[i][0]-1])
                    words[alignments[i][0] - 1]['src'].extend(hyp_src_word)
                    words[alignments[i][0] - 1]['trg'].append(trg_sent[alignments[i][1]-1])
                    i += 1
                elif next_op in ['n', 'i'] or i+1 >= len(alignments):
                    # Deal with string of deletions at start
                    if last_non_d_idx <= 0:
                        last_non_d_idx = i + 1
                    words[alignments[last_non_d_idx][0] - 1]['src'].extend(hyp_src_word)
                    i += 1
                else:
                    print(f'WARNING! Illegal operation: {next_op} in sentence {" ".join(src_sent)}')
                    print(f'{alignments[i]}')
                    print(f'{alignments[i+1]} <-- illegal operation')
                    i += 1
            
            # if the next operation is 'n' (i.e. source = target) or we are at
            #   the end of the sentence, attach the deleted token to the 
            #   previous non delete operation weather 's' or 'n'
            elif next_op in ['n', ''] :
                # Deal with string of deletions at start
                if last_non_d_idx <= 0:
                    last_non_d_idx = i + 1
                words[alignments[last_non_d_idx][0] - 1]['src'].append(src_sent[alignments[i][0] - 1])
                i +=1

        # a substitute followed by an insert, this means that the target token
        #   is aligned to nothing on the source side 
        # the following process is the same as a substitute followed by a delete
        elif current_op == 's' and next_op == 'i':
            last_s_idx = i
            last_non_d_op = 's'
            last_non_d_idx = i
            current_trg_word = trg_sent[alignments[i][1] - 1]
            current_src_word = src_sent[alignments[i][0] - 1]
            hyp_trg_word = [current_trg_word]
            current_dist = editdistance.distance(current_trg_word, current_src_word)

            continue_merge = False

            while next_op == 'i' and i < len(alignments) and continue_merge:
                i += 1
                temp_word = ''.join(hyp_trg_word) + trg_sent[alignments[i][1] - 1]
                new_dist = editdistance.distance(current_trg_word, temp_word)
                if new_dist >= current_dist:
                    i -= 1
                    continue_merge = False
                elif new_dist == 0:
                    hyp_trg_word.append(trg_sent[alignments[i][1] - 1])
                    continue_merge = False
                else:
                    continue_merge = True
                    current_dist = new_dist
                    hyp_trg_word.append(trg_sent[alignments[i][1] - 1])
                    if (i+1) < len(alignments):
                        next_op = alignments[i+1][2]
            if continue_merge:
                if temp_word.endswith(hyp_trg_word[-1]):
                    hyp_trg_word = hyp_trg_word[:-1]
                    i = i - 1
            words[alignments[current_idx][0] - 1]['trg'].extend(hyp_trg_word)
            words[alignments[current_idx][0] - 1]['src'].append(current_src_word)
            i += 1

        elif current_op == 'i':
            # try attaching it to the previous 's' and check if the edit distance decreases
            if i != 0 and last_s_idx != -1:
                prev_dist = editdistance.distance(''.join(words[alignments[last_s_idx][0] - 1]['trg']), ''.join(words[alignments[last_s_idx][0] - 1]['src']))
                
                hypo = words[alignments[last_s_idx][0] - 1]['trg'] + [trg_sent[alignments[i][1] - 1]]
                hyp_dist = editdistance.distance(''.join(words[alignments[last_s_idx][0] - 1]['src']), ''.join(hypo))
                if hyp_dist <= prev_dist:
                    words[alignments[last_s_idx][0] - 1]['trg'] = hypo
                    i += 1
                    continue

            # this is a i/d sequence, that is instead of a substitute it is 
            #   treated as an insert followed by a delete
            if next_op == 'd':
                words[alignments[i+1][0] - 1]['src'].append(src_sent[alignments[i+1][0]-1])
                words[alignments[i+1][0] - 1]['trg'].append(trg_sent[alignments[i][1]-1])
                last_non_d_op = 's'
                last_non_d_idx = i+1
                i += 2
            elif next_op == 's':
                i += 1
                last_s_idx = i
                last_non_d_op = 's'
                last_non_d_idx = i
                words[alignments[i][0] - 1]['trg'].append(trg_sent[alignments[i-1][1]-1])
                words[alignments[i][0] - 1]['trg'].append(trg_sent[alignments[i][1]-1])
                words[alignments[i][0] - 1]['src'].append(src_sent[alignments[i][0]-1])
                i += 1
            elif next_op == 'i':
                current_trg_word = trg_sent[alignments[i][1] - 1]
                hyp_trg_word = [current_trg_word]
                current_dist = editdistance.distance('', current_trg_word)
                done = False
                while next_op == 'i' and i+1 < len(alignments) and not done:
                    i += 1
                    temp_word = ''.join(hyp_trg_word) + trg_sent[alignments[i][1] - 1]
                    hyp_trg_word.append(trg_sent[alignments[i][1] - 1])
                    if (i+1) < len(alignments):
                        next_op = alignments[i+1][2]
                
                if next_op == 's':
                    i += 1
                    last_s_idx = i
                    last_non_d_op = 's'
                    last_non_d_idx = i
                    hyp_trg_word.append(trg_sent[alignments[i][1]-1])
                    words[alignments[i][0] - 1]['trg'].extend(hyp_trg_word)
                    words[alignments[i][0] - 1]['src'].append(src_sent[alignments[i][0]-1])
                    i += 1
                elif next_op in ['n', 'd'] or i+1 >= len(alignments):
                     # Deal with string of insertions at start
                    if last_non_d_idx <= 0:
                        last_non_d_idx = i + 1
                    words[alignments[last_non_d_idx][0] - 1]['trg'].extend(hyp_trg_word)
                    i +=1
                else:
                    print(f'WARNING! Illegal operation: {next_op} in sentence {" ".join(src_sent)}')
                    print(f'{alignments[i]}')
                    print(f'{alignments[i+1]} <-- illegal operation')
                    i += 1
            elif next_op in ['n', '']:
                # Deal with string of insertions at start
                if last_non_d_idx <= 0:
                    last_non_d_idx = i + 1
                words[alignments[last_non_d_idx][0] - 1]['trg'].append(trg_sent[alignments[i][1] - 1])
                i +=1
            else:
                print('WARNING: there might be a misalignment in sentence', ''.join(src_sent))
                i += 1

        # a substitute or no change
        elif current_op in ['n', 's']:
            words[alignments[i][0] - 1]['src'].append(src_sent[alignments[i][0] - 1])
            if current_op == 's':
                last_s_idx = i
                last_non_d_op = 's'
            else:
                last_non_d_op = 'n'
            
            last_non_d_idx = i
            words[alignments[i][0] - 1]['trg'].append(trg_sent[alignments[i][1] - 1])
            i += 1

    # remove empty placeholders
    for key in words:
        if words[key]['src'] == [] and words[key]['trg'] == []:
            continue
        if words[key]['src'] == []:
            words[key]['src'].append('NULL')
        elif words[key]['trg'] == []:
            words[key]['trg'].append('NULL')

        src_stream.write(f'{" ".join(words[key]["src"])}\n')
        trg_stream.write(f'{" ".join(words[key]["trg"])}\n')
        col_align_stream.write(f'{" ".join(words[key]["src"])}\t{" ".join(words[key]["trg"])}\n')

    src_stream.write('\n')
    trg_stream.write('\n')
    col_align_stream.write('\n')


def write_distances_only(distances, src_sent, trg_sent, file_stream):
    for distance in distances:
        if distance[0] is None:
            file_stream.write('\t<\t')
            file_stream.write(trg_sent[distance[1] - 1])
            file_stream.write(f'\t{distance}\n')
        elif distance[1] is None:
            file_stream.write(src_sent[distance[0] - 1])
            file_stream.write('\t>\t')
            file_stream.write(f'\t{distance}\n')
        else:
            file_stream.write(src_sent[distance[0] - 1])
            file_stream.write('\t')
            file_stream.write('=' if distance[2] == 'n' else '|')
            file_stream.write('\t')
            file_stream.write(trg_sent[distance[1] - 1])
            file_stream.write(f'\t{distance}\n')
    file_stream.write('\n')


if __name__ == "__main__":
    arguments = docopt(__doc__)
    src_sentences = open(arguments['--source'], 'r').readlines()
    trg_sentences = open(arguments['--target'], 'r').readlines()
    mode = arguments['--mode']

    if mode == 'align':
        src_output = open(arguments['--source']+'.align', 'w')
        trg_output = open(arguments['--target']+'.align', 'w')
        col_align_out = open(arguments['--output']+'.coAlign', 'w')
        col_align_out.write(f'SOURCE\tTARGET\n')
    elif mode == 'basic':
        output = open(arguments['--output']+'.basic', 'w')
    else:
        print(f"Warning: mode: [{arguments['--mode']}] is not valid, falling back to default mode: [align]")
        mode = 'align'
        src_output = open(arguments['--source']+'.align', 'w')
        trg_output = open(arguments['--target']+'.align', 'w')
        col_align_out = open(arguments['--output']+'.coAlign', 'w')
        col_align_out.write(f'SOURCE\tTARGET\n')

    for src, trg in zip(src_sentences, trg_sentences):
        alignments = align_words(src.strip(), trg.strip())
        src_sent = src.strip().split()
        trg_sent = trg.strip().split()

        if mode == 'align':
            write_exact_alignment_only(alignments, src_sent, trg_sent,
                                       src_output, trg_output,
                                       col_align_out)
        elif mode == 'basic':
            write_distances_only(alignments, src_sent, trg_sent, output)

    if mode == 'align':
        print(f'SOURCE alignments are saved to: {arguments["--source"]}.align')
        src_output.close()
        print(f'TARGET alignments are saved to: {arguments["--target"]}.align')
        trg_output.close()
        print(f'Side by side alignments are saved to: {arguments["--output"]}.coAlign')
        col_align_out.close()
    elif mode == 'basic':
        print(f'Basic alignments are saved to: {arguments["--output"]}.basic')
        output.close()
