from .model.seq2seq import Seq2Seq
from .utils.data_utils import Vectorizer
from .model.beam_decoder import BeamSampler
import json
import torch
import os

class NeuralRewriter:
    def __init__(self, model, vectorizer, beam_width, top_n_best, device):
        self.model = model
        self.vectorizer = vectorizer
        self.beam_decoder = BeamSampler(model=self.model,
                                 src_vocab_char=self.vectorizer.src_vocab_char,
                                 src_vocab_word=self.vectorizer.src_vocab_word,
                                 trg_vocab_char=self.vectorizer.trg_vocab_char,
                                 ged_tag_vocab=self.vectorizer.ged_tag_vocab,
                                 beam_width=beam_width,
                                 topk=top_n_best,
                                 device=device)


    @classmethod
    def from_pretrained(cls, model_path, use_gpu, beam_width=10, top_n_best=3):
        # loading the model's config
        with open(os.path.join(model_path, 'joint.config.json')) as f:
            model_config = json.load(f)

        model = Seq2Seq(**model_config)
        device = torch.device('cuda' if use_gpu and torch.cuda.is_available()
                              else 'cpu')
        # loading the model's state dict
        model.load_state_dict(torch.load(os.path.join(model_path, 'joint.pt'),
                                         map_location=device))
        model.eval()
        model = model.to(device)

        # loading the vectorizer
        with open(os.path.join(model_path,'vectorizer.json')) as f:
            vectorizer = Vectorizer.from_serializable(json.load(f))

        return cls(model, vectorizer, beam_width, top_n_best, device)

    def rewrite(self, token, ged_tag):
        """
        Uses the pretrained char-level seq2seq model to do the gender rewriting

        Args:
            - token (str): the src token.
            - ged_tag (str): the ged tag.

        Returns:
            - rewritten_tokens, proposals, proposed by (tuple)
        """
        # adding target gender (side constraint) to the token
        token_sc = f'<{ged_tag}>{token}'

        gender_alts = self.beam_decoder.beam_decode(token=token_sc,
                                                    add_side_constraints=True,
                                                    max_len=512)

        return gender_alts