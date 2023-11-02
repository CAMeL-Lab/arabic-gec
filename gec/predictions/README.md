## Grammatical Error Correction Outputs:

We provide the model outputs on the dev and test sets of [QALB-2014](qalb14), [QALB-2015](qalb15), and [ZAEBUC](zaebuc). Each directory has 5 subdirectories containing the outputs of the baselines (morph, chatgpt, mle, mle+morph) and the seq2seq models. We provide the outputs of all the seq2seq models we report on in our paper, including the GED granularity experiments.


Morph, ChatGPT, MLE, MLE+Morph, and the AraT5 variants were used to report results on the development sets only, whereas AraBART variants were used to report results on the development and test sets. 

1. `[qalb14|qalb15|zaebuc]_dev.preds.txt`:
2. `[qalb14|qalb15|zaebuc]_dev.preds.txt.pp`:
3. `[qalb14|qalb15|zaebuc]_dev.preds.txt.m2`:
4. `[[qalb14|qalb15|zaebuc]_test.preds.txt]`:
5. `[qalb14|qalb15|zaebuc]_test.preds.txt.nopnx]`:
6. `[qalb14|qalb15|zaebuc]_test.preds.txt.nopnx.m2]`:
7. `[qalb14|qalb15|zaebuc]_test.preds.txt.nopnx.official.m2]`:
