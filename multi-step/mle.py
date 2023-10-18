from data_utils import Dataset
from collections import defaultdict
import dill as pickle



def build_ngrams(sentence, pad_right=False, pad_left=False, ngrams=1):
    """
    Args:
     - sentence (list of str): a list of words.
     - ngrams (int): 2 for bigrams, 3 for trigrams, etc.
     - pad_right (bool): adding </s> to the end of sentence
     - pad_left (bool): adding <s> to the beginning of sentence
    Returns:
     - ngrams of the sentence (list of tuples)
    """

    if pad_right:
        sentence = sentence + ['</s>'] * (ngrams - 1)
    if pad_left:
        sentence = ['<s>'] * (ngrams - 1) + sentence
    return [tuple(sentence[i - (ngrams - 1): i + 1])
            for i in range(ngrams - 1, len(sentence))]

class CBR:
    """
    Corpus-based Rewriting
    to model P(target_word | source_word, ged_tag)"""

    def __init__(self, model, ngrams, backoff=True):
        self.model = model
        self.ngrams = ngrams
        self.backoff = backoff


    @classmethod
    def build_model(cls, dataset, ngrams=1, backoff=True, mode='full'):
        """
        Args:
            - dataset (Dataset obj)
            - backoff (bool): backoff to a lower order ngram during lookup.
            - ngrams (int): number of ngrams
        Returns:
            - cbr model (default dict): The cbr model where the
            keys are (source_word, ged_tag) and vals
            are target_word
        """

        model = defaultdict(lambda: defaultdict(lambda: 0))
        context = dict()

        for ex in dataset.examples:
            src_tokens = ex.src_tokens
            tgt_tokens = ex.tgt_tokens
            ged_tags = ex.ged_tags

            # getting counts of all ngrams
            # until ngrams == 1
            for i in range(ngrams):

                src_tokens_ngrams = build_ngrams(src_tokens, ngrams=i + 1,
                                                 pad_left=True)

                assert len(src_tokens) == len(src_tokens_ngrams)

                for j, tgt_w in enumerate(tgt_tokens):
                    ged_tag = ged_tags[j]
                    src_ngram = src_tokens_ngrams[j]

                    if is_model_tag(ged_tag, mode=mode):
                        # counts of (t_w, s_w, t_g)
                        model[(src_ngram, ged_tag)][tgt_w] += 1
                        # counts of (s_w, t_g)
                        context[(src_ngram, ged_tag)] = (1 +
                                 context.get((src_ngram, ged_tag), 0))

        # turning the counts into probs
        for sw, ged_tag in model:
            for tgt_w in model[(sw, ged_tag)]:
                model[(sw, ged_tag)][tgt_w] /= float(context[(sw, ged_tag)])

        return cls(model, ngrams, backoff)

    def __getitem__(self, sw_gt):
        context, ged_tag = sw_gt[0], sw_gt[1]

        if self.backoff:
            # keep backing-off until a context is found
            for i in range(self.ngrams):
                if (context[i:], ged_tag) in self.model:
                    return dict(self.model[(context[i:], ged_tag)])
        else:
            if (context, ged_tag) in self.model:
                return dict(self.model[(context, ged_tag)])
        # worst case, return None
        return None

    def __len__(self):
        return len(self.model)

    @staticmethod
    def load_model(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)


class CBR_WO_GED(CBR):
    def __init__(self, model, ngrams, backoff=True):
        super().__init__(model, ngrams, backoff)

    @classmethod
    def build_model(cls, dataset, ngrams=1, backoff=True):
        """
        Args:
            - dataset (Dataset obj)
            - backoff (bool): backoff to a lower order ngram during lookup.
            - ngrams (int): number of ngrams
        Returns:
            - cbr model (default dict): The cbr model where the
            keys are source_word and vals
            are target_word
        """

        model = defaultdict(lambda: defaultdict(lambda: 0))
        context = dict()

        for ex in dataset.examples:
            src_tokens = ex.src_tokens
            tgt_tokens = ex.tgt_tokens


            # getting counts of all ngrams
            # until ngrams == 1
            for i in range(ngrams):
                src_tokens_ngrams = build_ngrams(src_tokens, ngrams=i + 1,
                                                 pad_left=True)

                assert len(src_tokens) == len(src_tokens_ngrams)

                for j, tgt_w in enumerate(tgt_tokens):
                    src_ngram = src_tokens_ngrams[j]
                    # counts of (t_w, s_w)
                    model[src_ngram][tgt_w] += 1
                    # counts of (s_w)
                    context[src_ngram] = 1 + context.get(src_ngram, 0)

        # turning the counts into probs
        for sw in model:
            for tgt_w in model[sw]:
                model[sw][tgt_w] /= float(context[sw])

        return cls(model, ngrams, backoff)

    def __getitem__(self, sw_gt):
        context = sw_gt

        if self.backoff:
            # keep backing-off until a context is found
            for i in range(self.ngrams):
                if context[i:] in self.model:
                    return dict(self.model[context[i:]])
        else:
            if context in self.model:
                return dict(self.model[context])

        # worst case, return None
        return None


def is_model_tag(tag, mode='full'):
    if mode == 'full':
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
    else:
        tag_combs  = [
            'REPLACE_O+REPLACE_X',
            'REPLACE_M+REPLACE_O'
        ]

    if tag == 'UC' or tag == 'MERGE' or tag == 'UNK' or tag == 'DELETE':
        return False

    if '+' in tag and tag not in tag_combs:
        return False

    return True

if __name__ == '__main__':
    data = Dataset(raw_data_path='/scratch/ba63/gec/data/alignment/modeling_areta_tags_improved/qalb14/qalb14_train.areta+.nopnx.txt')
    model = CBR_WO_GED.build_model(data, ngrams=2)
    import pdb; pdb.set_trace()
    x = 10  