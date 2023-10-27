from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.morphology.database import MorphologyDB
from camel_tools.utils.dediac import dediac_ar
import argparse

import json


db = MorphologyDB('/scratch/ba63/gender-rewriting/models/calima-msa-s31_0.4.2.db')
analyzer = Analyzer(db)
bert_disambig = BERTUnfactoredDisambiguator.pretrained()
bert_disambig._analyzer = analyzer


def load_data(path):
    with open(path) as f:
        return [x.strip() for x in f.readlines()]

def correct(sentences):
    corrected_sents = []
    sents = [sent.split(' ') for sent in sentences]
    disambigs = bert_disambig.disambiguate_sentences(sents)

    for disambig in disambigs:
        corrected_sent = [dediac_ar(w_disambig.analyses[0].analysis['diac']) for w_disambig in disambig]
        corrected_sent = ' '.join(corrected_sent)
        corrected_sents.append(corrected_sent)

    return corrected_sents


def write_data(path, data):
    with open(path, mode='w') as f:
        for sent in data:
            f.write(sent)
            f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file')
    parser.add_argument('--output_file')
    args = parser.parse_args()

    input_data = load_data(args.input_file)
    corrected = correct(input_data)
    write_data(args.output_file, corrected)