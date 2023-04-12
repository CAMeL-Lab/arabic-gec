import logging
from error_identifier import ErrorIdentifier
from rules_tagger import GECTagger
from mle import CBR, build_ngrams
from data_utils import Dataset, postprocess_src_ged
from rules_tagger.rules_factory.rules import Rule
from rules_tagger.generate import detokenize
from neural_rewriter.rewriter import NeuralRewriter
from lm_rewriter.rewriter import LMRewriter
from aligner.aligner import align
from error_analysis import error_analysis, ErrorAnalysisExample
import re
import argparse


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Rewriter:
    def __init__(self, cbr_model, error_identifier=None, gec_tagger=None,
                neural_rewriter=None,
                lm_rewriter=None):

        self.cbr_model = cbr_model
        self.error_identifier = error_identifier
        self.gec_tagger = gec_tagger
        self.neural_rewriter = neural_rewriter
        self.lm_rewriter = lm_rewriter


    def rewrite(self, dataset, output_path, do_error_analysis=False):
        fixed_sents = []
        oov_cnt = 0
        err_words_cnt = 0

        error_anas = []

        for i, example in enumerate(dataset):
            src_tokens = example.src_tokens
            gold_ged_tags = example.ged_tags
            src_tokens, gold_ged_tags = postprocess_src_ged(src_tokens, gold_ged_tags)

            for token in src_tokens:
                assert len(token.split(' ')) == 1

            if self.error_identifier is not None:
                ged_tags = self.error_identifier.predict_sentence(src_tokens)
            else: # oracle experiments
                ged_tags = gold_ged_tags

            assert len(ged_tags) == len(src_tokens)

            tokens_ngrams = build_ngrams(src_tokens,
                                         ngrams=self.cbr_model.ngrams,
                                         pad_left=True)


            fixed_sent = []
            oov_indices = []
            fixed_oov_indices = []
            fixed_gen_src_indices = []
            gen_source = []

            # rewrting the erroneous words
            j = 0
            while j < len(src_tokens):
                if ged_tags[j] == 'UC':
                    gen_source.append('NA')
                    fixed_sent.append(src_tokens[j])
                    j += 1

                elif ged_tags[j] == 'DELETE':
                    gen_source.append('NA')
                    fixed_sent.append('')
                    err_words_cnt += 1
                    j += 1
                    continue

                elif ged_tags[j] == 'MERGE-B':
                    rewritten_token = []
                    rewritten_token.append(src_tokens[j])
                    err_words_cnt += 1
                    j += 1
                    gen_source.append('NA')

                    while  j < len(ged_tags) and 'MERGE-I' in ged_tags[j]:
                        err_words_cnt += 1
                        rewritten_token.append(src_tokens[j])
                        j += 1
                        gen_source.append('NA')

                    rewritten_token = ''.join(rewritten_token)
                    fixed_sent.append(rewritten_token)

                else:
                    err_words_cnt += 1
                    cbr_candidates = self.cbr_model[(tokens_ngrams[j],
                                                     ged_tags[j])]
                    if cbr_candidates:
                        rewritten_token = max(cbr_candidates.items(),
                                            key=lambda x: x[1])[0]

                        gen_source.append('CBR')
                        fixed_sent.append(rewritten_token)

                    else:

                        logger.info(f'OOV: {src_tokens[j], ged_tags[j]}')

                        oov_cnt += 1

                        fixed_oov_indices.append(len(fixed_sent))
                        fixed_gen_src_indices.append(len(gen_source))
                        if self.lm_rewriter is not None:
                            fixed_sent.append('OOV')
                        else:
                            fixed_sent.append(src_tokens[j])

                        gen_source.append('OOV')
                        oov_indices.append(j)
                    j += 1

            # fill-in oov words
            if len(oov_indices) != 0 and self.lm_rewriter is not None:
                rewritten_sent = self.lm_rewriter.rewrite(' '.join(src_tokens))
                alignment = align([' '.join(src_tokens)], [rewritten_sent])

                for oov_idx, fixed_oov_idx in zip (oov_indices, fixed_oov_indices):
                    fixed_sent[fixed_oov_idx] = alignment[oov_idx][1]
                    logger.info(f'Aligned Token: {alignment[oov_idx][1]}')

                for idx in fixed_gen_src_indices:
                    gen_source[idx] = 'T5'

            logger.info('\n')
            logger.info(f'{i}')
            logger.info(' '.join(src_tokens))

            logger.info('\n')

            fixed_sent = ' '.join(fixed_sent)

            error_anas.append(ErrorAnalysisExample(src_tokens=src_tokens,
                                                   generated_sent=fixed_sent,
                                                   generation_src=gen_source,
                                                   gold_ged_tags=gold_ged_tags,
                                                   pred_ged_tags=ged_tags,
                                                   )
                            )

            fixed_sent = re.sub(' +', ' ', fixed_sent)
            logger.info(fixed_sent)
            fixed_sents.append(fixed_sent)


        if do_error_analysis:
            error_analysis(error_anas, f'{output_path}.error_ana.txt')

        logger.info(f"Words with errors: {err_words_cnt}")
        logger.info(f"OOVs: {oov_cnt}")

        write_data(output_path, fixed_sents)

        # return fixed_sents


    def apply_rules(self, tagger_output, word_idx):
        word_ids = tagger_output['word_ids']
        subwords_model = [x for i, x in enumerate(tagger_output['subwords_model']) if word_ids[i] == word_idx]
        subwords = [x for i, x in enumerate(tagger_output['subwords']) if word_ids[i] == word_idx]
        rules = [x for i, x in enumerate(tagger_output['preds']) if word_ids[i] == word_idx]

        generated = []

        for rule, sw_model, sw in zip(rules, subwords_model, subwords):
            rule = Rule.from_str(rule)

            if sw_model.startswith('##'):
                generated.append(f"##{rule.apply(sw)}")

            elif sw.startswith(' '):
                generated.append(f" {rule.apply(sw)}")

            else:
                generated.append(rule.apply(sw))

        detokenize_generated = detokenize(generated)
        return detokenize_generated


def write_data(path, data):
    with open(path, mode='w') as f:
        f.write('\n'.join(data))
        f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file')
    parser.add_argument('--test_file')
    parser.add_argument('--cbr_ngrams', type=int)
    parser.add_argument('--ged_model')
    parser.add_argument('--seq2seq_model')
    parser.add_argument('--do_error_ana', action="store_true")
    parser.add_argument('--output_path')

    args = parser.parse_args()


    train_data = Dataset(raw_data_path=args.train_file)

    test_data = Dataset(raw_data_path=args.test_file)

    ged_model = None
    if args.ged_model:
        ged_model = ErrorIdentifier(model_path=args.ged_model)

    lm_rewriter = None
    if args.seq2seq_model:
        lm_rewriter = LMRewriter.from_pretrained(args.seq2seq_model, use_gpu=True)


    logger.info(f'Building the CBR model on {args.train_file}')

    cbr_model = CBR.build_model(train_data, ngrams=args.cbr_ngrams)


    rewriter = Rewriter(cbr_model=cbr_model,
                        error_identifier=ged_model,
                        gec_tagger=None,
                        lm_rewriter=lm_rewriter)


    rewritten_data = rewriter.rewrite(test_data,
                                      output_path=args.output_path,
                                      do_error_analysis=args.do_error_ana
                                     )


    # gec_tagger = GECTagger(model_path='/scratch/ba63/gec/models/rules_tagger/wo_camelira/10/checkpoint-1500')

    # train_data = Dataset(raw_data_path='/scratch/ba63/gec/data/gec_camelira/areta_tags/qalb14_train.areta+.txt')
    # tune_data = Dataset(raw_data_path='/scratch/ba63/gec/data/gec_camelira/areta_tags/qalb14_tune.areta+.txt')
    # error_identifier = ErrorIdentifier(model_path='/scratch/ba63/gec/models/ged/qalb14/w_camelira/checkpoint-1500')
    # gec_tagger = GECTagger(model_path='/scratch/ba63/gec/models/rules_tagger/w_camelira/10/checkpoint-4500')


    # neural_rewriter = NeuralRewriter.from_pretrained('neural_rewriter/saved_models', top_n_best=3, use_gpu=True)
    # lm_rewriter = LMRewriter.from_pretrained('/scratch/ba63/gec/models/gec/qalb14/t5_w_camelira', use_gpu=True)
