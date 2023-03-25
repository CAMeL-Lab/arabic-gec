import json
import copy


class InputExample:
    def __init__(self, src_tokens, tgt_tokens, areta_tags, morph_feats):
        self.src_tokens = src_tokens
        self.tgt_tokens = tgt_tokens
        self.areta_tags = areta_tags
        self.morph_feats = morph_feats

    def __repr__(self):
        return str(self.to_json_str())

    def to_json_str(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output


class Dataset:
    def __init__(self, raw_data_path, morph_feats_path=None):
        self.examples = self.create_examples(raw_data_path, morph_feats_path)

    def create_examples(self, raw_data_path, morph_feats_path):
        # loading morph features
        if morph_feats_path:
            with open(morph_feats_path) as f:
                morph_feats = [json.loads(line) for line in f.readlines()]
        else:
            morph_feats = None

        examples = []
        src_tokens = []
        tgt_tokens = []
        areta_tags = []

        morph_idx = 0

        with open(raw_data_path) as f:
            for i, line in enumerate(f.readlines()):
                line = line.split('\t')
                if len(line) == 3:
                    src, tgt, tag = line

                    src_tokens.append(src.strip())
                    tgt_tokens.append(tgt.strip())
                    areta_tags.append(tag.strip())

                else:
                    ex_morph_feats = morph_feats[morph_idx]['feats'] if morph_feats else None
                    examples.append(InputExample(src_tokens=src_tokens,
                                                 tgt_tokens=tgt_tokens,
                                                 areta_tags=areta_tags,
                                                 morph_feats=ex_morph_feats)
                                    )

                    src_tokens, tgt_tokens, areta_tags = [], [], []
                    morph_idx += 1

            if src_tokens and tgt_tokens and areta_tags:
                ex_morph_feats = morph_feats[morph_idx]['feats'] if morph_feats else None
                examples.append(InputExample(src_tokens=src_tokens,
                                                 tgt_tokens=src_tokens,
                                                 areta_tags=areta_tags,
                                                 morph_feats=ex_morph_feats)
                                            )
        return examples

    def __len__(self):
        return len(self.examples)

# def load_raw_data(path):
#     src_tokens = []
#     tgt_tokens = []
#     tags = []
#     examples = []

#     with open(path, mode='r') as f:
#         for line in f.readlines():
#             line = line.split('\t')
#             if len(line) == 3:
#                 src, tgt, tag = line
#                 src_tokens.append(src)
#                 tgt_tokens.append(tgt)
#                 tags.append(tag.strip())

#             else:
#                 examples.append((src_tokens, tgt_tokens, tags))
#                 src_tokens = []
#                 tgt_tokens = []
#                 tags = []

#         if src_tokens and tgt_tokens and tags:
#             examples.append((src_tokens, tgt_tokens, tags))

#     return examples

# def load_morph_features(path):
#     data = []
#     with open(path, mode='r') as f:
#         for line in f.readlines():
#             data.append(json.loads(line))

#     return data

# def load_data(raw_data_path, morph_feats_path):
#     raw_data = load_raw_data(raw_data_path)
#     morph_feats = load_morph_features(morph_feats_path)

if __name__ == '__main__':
    data = Dataset(raw_data_path='/scratch/ba63/gec/data/alignment/modeling_areta_tags_check/qalb14/corruption_data/qalb14_tune.areta.txt',
                   morph_feats_path='/scratch/ba63/gec/data/alignment/modeling_areta_tags_check/qalb14/corruption_data/tune_morph.json')
    import pdb; pdb.set_trace()
    x = 10