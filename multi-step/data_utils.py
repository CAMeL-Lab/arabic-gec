import json
import copy


class InputExample:
    def __init__(self, src_tokens, tgt_tokens, ged_tags):
        self.src_tokens = src_tokens
        self.tgt_tokens = tgt_tokens
        self.ged_tags = ged_tags

    def __repr__(self):
        return str(self.to_json_str())

    def to_json_str(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output


class Dataset:
    def __init__(self, raw_data_path):
        self.examples = self.create_examples(raw_data_path)

    def create_examples(self, raw_data_path):

        examples = []
        src_tokens = []
        tgt_tokens = []
        ged_tags = []


        with open(raw_data_path) as f:
            for i, line in enumerate(f.readlines()[1:]):
                line = line.split('\t')
                if len(line) == 3:
                    src, tgt, tag = line

                    # exclude insertions
                    if 'INSERT' in tag:
                        continue

                    src_tokens.append(src.strip())
                    tgt_tokens.append(tgt.strip())
                    ged_tags.append(tag.strip())

                else:

                    examples.append(InputExample(src_tokens=src_tokens,
                                                 tgt_tokens=tgt_tokens,
                                                 ged_tags=ged_tags)
                                    )

                    src_tokens, tgt_tokens, ged_tags = [], [], []


            if src_tokens and tgt_tokens and ged_tags:

                examples.append(InputExample(src_tokens=src_tokens,
                                             tgt_tokens=src_tokens,
                                             ged_tags=ged_tags)
                                )
        return examples


    def __getitem__(self, idx):
        return self.examples[idx]


    def __len__(self):
        return len(self.examples)


def postprocess_src_ged(src_tokens, ged_tags):
    """
    1) Replaces out-of-set tag combinations with UNK.
    2) Projects a tag on a span of tokens.
    This is important because we need to deal with a single source
    token at a time.
    """
    tag_combs = [
            'REPLACE_OH+REPLACE_OM',
            'REPLACE_OH+REPLACE_OT',
            'REPLACE_OD+REPLACE_OR',
            'REPLACE_OD+REPLACE_OG',
            'REPLACE_XC+REPLACE_XN',
            'REPLACE_OA+REPLACE_OH',
            'REPLACE_OM+REPLACE_OR',
            'REPLACE_OH+REPLACE_XC',
            'REPLACE_OD+REPLACE_OH',
            'REPLACE_XC+REPLACE_XG',
            'REPLACE_MI+REPLACE_OH',
            'REPLACE_OA+REPLACE_OR',
            'REPLACE_OR+REPLACE_OT',
            'REPLACE_OD+REPLACE_OM'
            ]

    assert len(src_tokens) == len(ged_tags)
    ged_tags_ = []
    src_tokens_ = []

    for token, tag in zip(src_tokens, ged_tags):
        if ('+' in tag and tag not in tag_combs) or (tag == 'UNK'):
            tag = 'UNK'

        words, tags = project_span(token, tag)

        src_tokens_ += words
        ged_tags_ += tags

    return src_tokens_, ged_tags_


def project_span(word, tag):
    if 'MERGE' not in tag:
        return word.split(), [tag for _ in range(len(word.split()))]

    return word.split(), ['MERGE-B']+['MERGE-I' for _ in range(len(word.split()) - 1)]



