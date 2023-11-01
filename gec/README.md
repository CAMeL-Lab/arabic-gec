# Grammatical Error Correction

We follow the same constraints that were introduced in the QALB-2014 and QALB-2015 shared tasks: systems tested on QALB-2014 are trained only on QALB-2014 training data, whereas systems tested on QALB-2015 are trained on QALB-2014 and QALB-2015 training data. For ZAEBUC, we train our systems on the combination of the three training datasets. The data used to fine-tune all GEC model variants is [here](https://github.com/balhafni/arabic-gec/tree/master/data/gec).<br/>


## Baselines:

### Morphological Disambiguation:
We use the contextual morphological analysis and disambiguation model introduced by [Inoue et al., 2022]() which is publicly available in [CAMeL Tools](). This step is done by running the [camelira_gec.sh]() script. 


### MLE:
The code for the MLE model we describe in our paper can be found [here](mle), with instructions on how to run it.


### GPT-3.5:
We provide the code we used to prompt gpt-3.5 turbo (ChatGPT) along the full set of prompts [here](chatgpt_gec.py). Make sure to provide your OpenAI **[API_KEY](https://github.com/CAMeL-Lab/arabic-gec/blob/master/gec/chatgpt_gec.py#L12)** before running the script.


## Seq2Seq Models:


### Fine-tuning:


### Evaluation:

Talk about the m2scorer skip time on the dev set.

If the file name has a pp, sentences were skipped. we do this only on the dev sets.

If the file name has official, this is the python2 m2scorer from the qalb release.


At the end of the fine-tuning, we pick the best checkpoint based on the F<sub>0.5</sub> performance on the dev sets.
