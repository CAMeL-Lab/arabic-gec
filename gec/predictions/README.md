## Grammatical Error Correction Outputs:

We provide the model outputs on the dev and test sets of [QALB-2014](qalb14), [QALB-2015](qalb15), and [ZAEBUC](zaebuc). Each directory has 5 subdirectories containing the outputs of the baselines (morph, chatgpt, mle, mle+morph) and the seq2seq models. We provide the outputs of all the seq2seq models we report on in our paper, including the GED granularity experiments.


Morph, ChatGPT, MLE, MLE+Morph, and the AraT5 variants were used to report results on the development sets only, whereas AraBART variants were used to report results on the development and test sets. Each of the subfolders in each directory could potentially has the following files based on the experiments we conduct:

1. `[qalb14|qalb15|zaebuc]_dev.preds.txt`: the models' outputs  on the development sets.
2. `[qalb14|qalb15|zaebuc]_dev.preds.txt.pp`: the models' outputs  of the development sets after replacing the generated sentences that differ significantly from the input by the input sentences.
3. `[qalb14|qalb15|zaebuc]_dev.preds.txt.m2`: the m2scorer evaluation of the `[qalb14|qalb15|zaebuc]_dev.preds.txt` files.
4. `[qalb14|qalb15|zaebuc]_dev.preds.txt.pp.m2`: the m2scorer evaluation of the `[qalb14|qalb15|zaebuc]_dev.preds.txt.pp` files.
5. `[qalb14|qalb15|zaebuc]_test.preds.txt]`: the models' outputs on the test sets.
6. `[qalb14|qalb15|zaebuc]_test.preds.txt.nopnx]`: the models' no-punctuation outputs on the test sets.
7. `[qalb14|qalb15|zaebuc]_test.preds.txt.m2]`: the m2scorer evaluation of the `[qalb14|qalb15|zaebuc]_test.preds.txt]` files
8. `[qalb14|qalb15|zaebuc]_test.preds.txt.nopnx.m2]`: the m2scorer evaluation of the `[qalb14|qalb15|zaebuc]_test.preds.txt.nopnx]` files
