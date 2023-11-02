# Grammatical Error Correction

We follow the same constraints that were introduced in the QALB-2014 and QALB-2015 shared tasks: systems tested on QALB-2014 are trained only on QALB-2014 training data, whereas systems tested on QALB-2015 are trained on QALB-2014 and QALB-2015 training data. For ZAEBUC, we train our systems on the combination of the three training datasets. The data used to fine-tune all GEC model variants is [here](https://github.com/balhafni/arabic-gec/tree/master/data/gec).<br/>


## Baselines:

### Morphological Disambiguation:
We use the contextual morphological analysis and disambiguation model introduced by [Inoue et al., 2022]() which is publicly available in [CAMeL Tools](). This step is done by running the [camelira_gec.sh]() script. 


### MLE:
The code for the MLE model we describe in our paper can be found [here](mle), with instructions on how to run it.


### GPT-3.5:
We provide the code we used to prompt gpt-3.5 turbo (ChatGPT) along the full set of prompts [here](chatgpt_gec.py). Make sure to provide your OpenAI [API_KEY](https://github.com/CAMeL-Lab/arabic-gec/blob/master/gec/chatgpt_gec.py#L12) before running the script.


## Seq2Seq Models:

We provide [scripts](training_scripts) to reproduce the GEC models we built by fine-tuning [AraBART]() and [AraT5](). It is important to note that you need to specify the correct training file corresponding to each experiment. We provide a detailed description of the data we used to build our GEC models [here]((https://github.com/CAMeL-Lab/arabic-gec/tree/master/data)). Each of the provided [scripts](training_scripts] will have a variant of the following code based on the experiment we'd like to run:

```bash

MODEL=/path/to/model # or huggingface model id
OUTPUT_DIR=/path/to/output/dir
TRAIN_FILE=/path/to/gec/train/file
LABELS=/path/to/ged/labels
STEPS=1000 # 500 for qalb14 and 1000 for mix
BATCH_SIZE=16 # 32 for qalb14 and 16 for mix

python run_gec.py \
  --model_name_or_path $MODEL \
  --do_train \
  --optim adamw_torch \
  --source_lang raw \
  --target_lang cor \
  --train_file $TRAIN_FILE \
  --ged_tags  $LABELS \
  --save_steps $STEPS \
  --num_train_epochs 10 \
  --output_dir $OUTPUT_DIR \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --max_target_length 1024 \
  --seed 42 \
  --overwrite_cache \
  --overwrite_output_dir
```

We also provide the [scripts](generation_scripts) we used during inference to generate grammatically correct outputs using the fine-tuned models. Each of the provided [scripts](generation_scripts) will have a variant of the following code based on the experiment we'd like to run:

```bash

model=/path/to/model
pred_file=/prediction/file/name
test_file=/path/to/test/file # dev or test 
m2_edits=/path/to/m2edits
m2_edits_nopnx=/path/to/m2edits/without/punctuation

python generate.py \
  --model_name_or_path $model \
  --source_lang raw \
  --target_lang cor \
  --use_ged \
  --preprocess_merges \
  --test_file $test_file \
  --m2_edits $m2_edits \
  --m2_edits_nopnx $m2_edits_nopnx \
  --per_device_eval_batch_size 16 \
  --output_dir $checkpoint \
  --num_beams 5 \
  --num_return_sequences 1 \
  --max_target_length 1024 \
  --predict_with_generate \
  --prediction_file $pred_file
```

The generated outputs of our models on the dev and test sets are in the [predictions](predictions) directory.

## Evaluation:

We pick the best model checkpoint (for the seq2seq models) based on the development sets. However, the M<sup>2</sup> scorer suffers from extreme running times in cases where the generated outputs differ significantly from the input. To mitigate this bottleneck, we extend the [M<sup>2</sup> scorer](https://github.com/CAMeL-Lab/arabic-gec/tree/master/gec/utils/m2scorer) by introducing a python3 version with a [time limit](https://github.com/CAMeL-Lab/arabic-gec/blob/master/gec/utils/m2scorer/m2scorer.py#L141) for each sentence during evaluation. If the evaluation of a single generated sentence surpasses this limit, we pass the input sentence to the output without modifications.
We implement this extension in a modular way that allows us to import and use M<sup>2</sup> scorer to evaluate the generated outputs within other scripts. In cases where the generated sentences are replaced by the input, the generated output file will end with a `.pp` extension. Note that our version of the  M<sup>2</sup> scorer has been tested and is identical to the M<sup>2</sup> scorer released as part of the QALB shared task in terms of evaluation.


It is important to note that we use our extended version of the M<sup>2</sup> scorer when reporting our results on the development sets **only**. When reporting our results on the
test sets, we use M<sup>2</sup> scorer python2 release that is provided by the QALB shared task. During the evaluation, we use the manually created m2edits for QALB-2014 and QALB-2015 which are publicly available as part of the shared task. For ZAEBUC, we rely on our alignment to create the m2edits automatically. For the no-punctuation evaluation, we use the automatically created m2edits for all datasets.


Running our extended version of the M<sup>2</sup> scorer can be done using the [eval_checkpoint_skip.sh](eval_checkpoint_skip.sh) script, whereas running the QALB shared task  M<sup>2</sup> scorer can be done using the [eval_checkpoint.sh](eval_checkpoint.sh) script.


## Error Analysis:

We provide a detailed [error analysis](error_analysis) of the specific error type performance of AraBART against our best system (i.e., AraBART+Morph+GED<sup>13</sup>) on average on the development sets. 

