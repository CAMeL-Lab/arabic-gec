import json
import copy
import re

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

                    # exclude inserts
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
