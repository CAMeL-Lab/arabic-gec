# Grammatical Error Correction

We follow the same constraints that were introduced in the QALB-2014 and QALB-2015 shared tasks: systems tested on QALB-2014 are trained only on QALB-2014 training data, whereas systems tested on QALB-2015 are trained on QALB-2014 and QALB-2015 training data. For ZAEBUC, we train our systems on the combination of the three training datasets. The data used to fine-tune all GEC model variants is [here](https://github.com/balhafni/arabic-gec/tree/master/data/gec).<br/>

At the end of the fine-tuning, we pick the best checkpoint based on the F<sub>0.5</sub> performance on the dev sets.

## Fine-tuning:




## Evaluation:
