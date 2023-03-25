from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.morphology.database import MorphologyDB
import argparse
import json


db = MorphologyDB('/scratch/ba63/gender-rewriting/models/calima-msa-s31_0.4.2.db')
analyzer = Analyzer(db)
bert_disambig = BERTUnfactoredDisambiguator.pretrained()
bert_disambig._analyzer = analyzer


def load_data(path):
    tgt_tokens = []
    examples = []

    with open(path, mode='r') as f:
        for line in f.readlines():
            line = line.split('\t')
            if len(line) == 3:
                src, tgt, tag = line
                tgt_tokens.append(tgt)
            else:
                examples.append(tgt_tokens)
                tgt_tokens = []

        if tgt_tokens:
            examples.append(tgt_tokens)

    return examples


def tag_sentences(sentences):
    all_morph_features = []

    for i, sentence in enumerate(sentences):
        if i % 1000 == 0:
            print(i, flush=True)

        indices = []
        words = []

        for j, word in enumerate(sentence):
            if word:
                indices.extend(j for x in word.split())
                words.extend(w for w in word.split())
            else:
                indices.append(j)
                words.append(word)

        disambig = bert_disambig.disambiguate(words)

        words_features = []
        ana_idx, word_idx = 0, 0

        while word_idx < len(words):
            if words[word_idx]:
                morph_features = get_morph_features(disambig[ana_idx])
                words_features.append(morph_features)
                word_idx += 1
                ana_idx += 1
            else:
                words_features.append(None)
                word_idx += 1

        assert len(words_features) == len(words) == len(indices)


        # collapsing the features for multiple words in the target
        collapsed_feats = []

        for idx, word, features in zip(indices, words, words_features):
            if len(collapsed_feats) > idx and collapsed_feats[idx]:
                collapsed_feats[idx].append(features)
            else:
                collapsed_feats.append([features])

        assert len(collapsed_feats) == len(sentence)

        all_morph_features.append({'words': sentence, 'feats': collapsed_feats})


    return all_morph_features


def get_morph_features(word_disambig):
    analysis = word_disambig.analyses[0]
    morph_feats = {}
    features = ['pos', 'gen', 'cas', 'num', 'stt', 'prc0', 'prc1',
                'prc2', 'prc3', 'enc0']
    for feature in features:
        if feature not in analysis.analysis:
            morph_feats[feature] = 'na'
        else:
            morph_feats[feature] = analysis.analysis[feature] if analysis.analysis[feature] != '-' else 'na'
    return morph_feats


def write_data(path, data):
    with open(path, mode='w') as f:
        for ex in data:
            f.write(json.dumps(ex, ensure_ascii=False))
            f.write('\n')

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')
    
    args = parser.parse_args()
    
    data = load_data(args.input)
    tagged_data = tag_sentences(data)
    
    write_data(args.output, tagged_data)
    
    # train_data = load_data('/scratch/ba63/gec/data/alignment/modeling_areta_tags_check/'\
    #                       'qalb14/corruption_data/qalb14_train.areta.txt')

    # dev_data = load_data('/scratch/ba63/gec/data/alignment/modeling_areta_tags_check/'\
    #                      'qalb14/corruption_data/qalb14_tune.areta.txt')

    #     train_data = load_data('/scratch/ba63/gec/data/alignment/modeling_areta_tags_check/'\
    #                           'qalb14/corruption_data/zaebuc_train.areta.txt')

    #     dev_data = load_data('/scratch/ba63/gec/data/alignment/modeling_areta_tags_check/'\
    #                          'qalb14/corruption_data/zaebuc_dev.areta.txt')

    #     train_tagged = tag_sentences(train_data)
    #     dev_tagged = tag_sentences(dev_data)

    # write_data('/scratch/ba63/gec/data/alignment/modeling_areta_tags_check/'\
    #            'qalb14/corruption_data/qalb14_train_morph.json', train_tagged)

    # write_data('/scratch/ba63/gec/data/alignment/modeling_areta_tags_check/'\
    #             'qalb14/corruption_data/qalb14_tune_morph.json', dev_tagged)

#     write_data('/scratch/ba63/gec/data/alignment/modeling_areta_tags_check/'\
#                'qalb14/corruption_data/zaebuc_train_morph.json', train_tagged)

#     write_data('/scratch/ba63/gec/data/alignment/modeling_areta_tags_check/'\
#                 'qalb14/corruption_data/zaebuc_dev_morph.json', dev_tagged)
