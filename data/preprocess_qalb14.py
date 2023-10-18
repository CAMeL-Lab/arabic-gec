from camel_tools.utils.charsets import (
    AR_LETTERS_CHARSET,
    UNICODE_PUNCT_SYMBOL_CHARSET)
import re
import string
import argparse


PUNCS = string.punctuation + ''.join(list(UNICODE_PUNCT_SYMBOL_CHARSET))

def read_data(path):
    with open(path, mode='r', encoding='utf8') as f:
        return [line.strip() for line in f.readlines()]

def get_spans(sentence):
    """
    Gets spans containing words of length 4 and less.
    It ensures that the span has at least 1 single Arabic letter in it.
    """
    count_chars = 0
    sentence_ = sentence.split()
    indices = []
    spans = []
    single_char_count = 0

    for i, word in enumerate(sentence_):
        # if the word is 4 characters or less long,
        # attemp to construct a span
        if (len(word) <= 3) and word != '.':
            if len(word) == 1 and word in AR_LETTERS_CHARSET:
                single_char_count += 1
            count_chars += 1
            indices.append(i)
        else:
            # if we have seen at least 3 single characters in the
            # span, add the span to the list
            if single_char_count >= 3:
                spans.append((count_chars, indices))
                single_char_count = 0

            count_chars = 0
            indices = []

    if spans:
        return spans
    else:
        return []

def fix_sentence(sentence, spans):
    """
    Given a buffy sentence and a list of spans,
    reconstruct the sentence by stitching the words in
    each span together
    """
    sentence = sentence.split()
    new_sentence = []
    start = 0

    for span in spans:
        span_start, span_end = span[1][0], span[1][-1]
        end = span_start

        # add words outside of span to the new sentence
        new_sentence += sentence[start: end]

        start = span_end + 1

        stitched = []

        # stitch words in a given span
        for word in sentence[span_start: span_end + 1]:

            if re.search(r'[' +"".join(AR_LETTERS_CHARSET)+ ']+', word):
                stitched.append(word)

            elif word != ' ':
                stitched.append(f'{word}')

        stitched = "".join(stitched)
        # separating puncs
        stitched = re.sub(r'([' + re.escape(PUNCS) + '])', r' \1 ',
                           stitched).strip()
        new_sentence += ["".join(stitched)]

    # add remaining parts of the sentence and remove double spaces
    new_sentence = " ".join(new_sentence).strip() + " " + " ".join(sentence[span_end + 1:])
    return re.sub(' +', ' ', new_sentence).strip()

def fix_sentences(lines):
    """
    Gets all the buggy sentence based on a heuristic and fix them
    """
    fixed_sents_cnt = 0
    clean_sentences = []
    for i, line in enumerate(lines):
        spans = get_spans(line)
        if spans:
            # get the max span length
            max_span = max(spans, key=lambda x: x[0])
            # if a span has more than 25 characters
            # then its buggy. This seems to cover all 
            # cases in train, dev, and test of QALB 2014
            if max_span and len(max_span[1]) > 25:
                fixed_sentence = fix_sentence(line, spans)
                clean_sentences.append(fixed_sentence)
                fixed_sents_cnt += 1
            else:
                clean_sentences.append(line)
        else:
            clean_sentences.append(line)
    print(f'{fixed_sents_cnt} sentences have been fixed!')
    return clean_sentences

def write_data(data, path):
    with open(path, mode='w', encoding='utf8') as f:
        for line in data:
            f.write(line)
            f.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', help='Input file.')
    parser.add_argument('--output_file', help='Output file.')
    args = parser.parse_args()

    data = read_data(args.input_file)
    print(f'Read {len(data)} sentences!')
    fixed_data = fix_sentences(data)
    write_data(fixed_data, args.output_file)
