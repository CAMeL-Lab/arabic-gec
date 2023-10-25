from aligner.aligner import align_error_analysis
from data_utils import project_span


class ErrorAnalysisExample:
    def __init__(self, src_tokens, generated_sent, gold_ged_tags, pred_ged_tags,
                generation_src):

        self.src_tokens = src_tokens
        self.generated_sent = generated_sent
        self.gold_ged_tags = gold_ged_tags
        self.pred_ged_tags = pred_ged_tags
        self.generation_src = generation_src


def error_analysis(error_anas, path):

    with open(path, 'w') as f:
        f.write('SRC\tGEN\tGOLD_TAG\tPRED_TAG\tSYS\n')
        for ex in error_anas:
            src_tokens = ex.src_tokens
            generated_sent = ex.generated_sent
            gold_tags = ex.gold_ged_tags
            preds_tags = ex.pred_ged_tags
            gen_source = ex.generation_src

            src_gen_align = align_error_analysis([' '.join(src_tokens)], [generated_sent])

            assert len(gold_tags) == len(preds_tags) == len(gen_source)

            ged_tag_idx = 0

            # projecting multiple source tokens
            for i, src_gen in enumerate(src_gen_align):
                src = src_gen[0].split()
                gen = src_gen[1]

                gold_tags_ = []
                pred_tags_ = []
                gen_source_ = []

                for i, word in enumerate(src):
                    if i == 0:
                        # print(f'{word}\t{gen}\t{gold_tags[ged_tag_idx]}\t{preds_tags[ged_tag_idx]}\t{gen_source[ged_tag_idx]}')
                        f.write(f'<s>{word}<s>\t<s>{gen}<s>\t{gold_tags[ged_tag_idx]}\t{preds_tags[ged_tag_idx]}\t{gen_source[ged_tag_idx]}\n')
                    else:
                        # print(f'{word}\t\t{gold_tags[ged_tag_idx]}\t{preds_tags[ged_tag_idx]}\t{gen_source[ged_tag_idx]}')
                        f.write(f'<s>{word}<s>\t<s><s>\t{gold_tags[ged_tag_idx]}\t{preds_tags[ged_tag_idx]}\t{gen_source[ged_tag_idx]}\n')
                    gold_tags_.append(gold_tags[ged_tag_idx])
                    pred_tags_.append(preds_tags[ged_tag_idx])
                    gen_source_.append(gen_source[ged_tag_idx])

                    ged_tag_idx += 1

            f.write('\n')
