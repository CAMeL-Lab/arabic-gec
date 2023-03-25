from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.morphology.database import MorphologyDB
from camel_tools.utils.dediac import dediac_ar

import json


db = MorphologyDB('/scratch/ba63/gender-rewriting/models/calima-msa-s31_0.4.2.db')
analyzer = Analyzer(db)
bert_disambig = BERTUnfactoredDisambiguator.pretrained()
bert_disambig._analyzer = analyzer


def load_data(path):
    src_tokens = []
    examples = []

    with open(path, mode='r') as f:
        for line in f.readlines()[1:]:
            line = line.split('\t')
            if len(line) == 2:
                src, tgt = line
                src_tokens.append(src)
            else:
                examples.append(src_tokens)
                src_tokens = []

        if src_tokens:
            examples.append(src_tokens)

    return examples


def correct(sentences):
    all_corrected_sentences = []

    for i, sentence in enumerate(sentences):
        if i % 1000 == 0:
            print(i, flush=True)

        words = []
        space_idx = []
        indices = []
        split_idx = 0

        for j, word in enumerate(sentence):
            if word.strip():
                indices.extend(split_idx for x in word.split())
                words.extend(w for w in word.split())
                split_idx += 1
            else:
                space_idx.append(j)

        disambig = bert_disambig.disambiguate(words)

        corrected = []
        for w_disambig in disambig:
            analysis = w_disambig.analyses[0].analysis
            correction = dediac_ar(analysis['diac'])
            corrected.append(correction)

        # if len(corrected) != len(sentence):
        #     import pdb; pdb.set_trace()

        collapsed_corrected = []
        # collapsing the split cases
        for idx, correct_word in zip(indices, corrected):
            if len(collapsed_corrected) > idx:
                collapsed_corrected[idx].append(correct_word)
            else:
                collapsed_corrected.append([correct_word])

        collapsed_corrected = [" ".join(sublist) for sublist in collapsed_corrected]

        # inserting back the empty spaces
        for idx in space_idx:
            collapsed_corrected.insert(idx, '')

        # import pdb; pdb.set_trace()

        assert len(collapsed_corrected) == len(sentence)

        all_corrected_sentences.append(collapsed_corrected)

    return all_corrected_sentences


def write_data(path, data):
    with open(path, mode='w') as f:
        for sent in data:
            for token in sent:
                f.write(token)
                f.write('\n')
            f.write('\n')


train_data = load_data('/scratch/ba63/gec/data/alignment/modeling_alignment/'\
                      'qalb14/qalb14_train.txt')
train_corrected = correct(train_data)
write_data('/scratch/ba63/gec/data/alignment/data_analysis_alignment/'\
            'improved_alignment/camelira_src_output_check/qalb14_train.src_camelbert.txt', train_corrected)


dev_data = load_data('/scratch/ba63/gec/data/alignment/modeling_alignment/'\
                      'qalb14/qalb14_tune.txt')
dev_corrected = correct(dev_data)
write_data('/scratch/ba63/gec/data/alignment/data_analysis_alignment/'\
            'improved_alignment/camelira_src_output_check/qalb14_tune.src_camelbert.txt', dev_corrected)



train_data = load_data('/scratch/ba63/gec/data/alignment/modeling_alignment/'\
                      'qalb15/qalb15_train.txt')
train_corrected = correct(train_data)
write_data('/scratch/ba63/gec/data/alignment/data_analysis_alignment/'\
            'improved_alignment/camelira_src_output_check/qalb15_train.src_camelbert.txt', train_corrected)


dev_data = load_data('/scratch/ba63/gec/data/alignment/modeling_alignment/'\
                      'qalb15/qalb15_dev.txt')
dev_corrected = correct(dev_data)
write_data('/scratch/ba63/gec/data/alignment/data_analysis_alignment/'\
            'improved_alignment/camelira_src_output_check/qalb15_dev.src_camelbert.txt', dev_corrected)



train_data = load_data('/scratch/ba63/gec/data/alignment/modeling_alignment/'\
                      'zaebuc/zaebuc_train.txt')
train_corrected = correct(train_data)
write_data('/scratch/ba63/gec/data/alignment/data_analysis_alignment/'\
            'improved_alignment/camelira_src_output_check/zaebuc_train.src_camelbert.txt', train_corrected)


dev_data = load_data('/scratch/ba63/gec/data/alignment/modeling_alignment/'\
                      'zaebuc/zaebuc_dev.txt')
dev_corrected = correct(dev_data)
write_data('/scratch/ba63/gec/data/alignment/data_analysis_alignment/'\
            'improved_alignment/camelira_src_output_check/zaebuc_dev.src_camelbert.txt', dev_corrected)
