# Grammatical Error Detection

All of our grammatical error detection (GED) models were built by fine-tuning [CAMeLBERT MSA](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-msa).

## Fine-tuning:

We follow the same constraints that were introduced in the QALB-2014 and QALB-2015 shared tasks: systems tested on QALB-2014 are only allowed to use the QALB-2014 training data, whereas systems tested on QALB-2015 are allowed to use the QALB-2014 and QALB-2015 training data. For ZAEBUC, we train our systems on the combinations of the three training datasets. The data used to fine-tune CAMeLBERT for all GED model variants is [here](https://github.com/balhafni/arabic-gec/tree/master/data/ged).<br/>

At the end of the fine-tuning, we pick the best checkpoint based on the F<sub>0.5</sub> performance of the GED task on the dev sets.


**TODO**: The fine-tuned GED CAMeLBERT models, which we use with our best GEC system, are available here this.


To run the fine-tuning:

```bash
export DATA_DIR=/path/to/data
export BERT_MODEL=/path/to/pretrained_model/ # Or huggingface model id 
export OUTPUT_DIR=/path/to/output_dir
export BATCH_SIZE=32
export NUM_EPOCHS=10
export SAVE_STEPS=500
export SEED=42

python error_detection.py \
    --data_dir $DATA_DIR \
    --optim adamw_torch \
    --labels $DATA_DIR/labels.txt \
    --model_name_or_path $BERT_MODEL \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --save_steps $SAVE_STEPS \
    --seed $SEED \
    --do_train \
    --overwrite_output_dir
```


## Inference:
To run inference:

```bash
export DATASET=qalb14 # or qalb15 or zaebuc
export exp=qalb14 # or qalb14-15 or mix
export DATA_DIR=/path/to/data
export OUTPUT_DIR=/path/to/model
export LABELS=/path/to/labels
export BATCH_SIZE=32
export SEED=42
export pred_mode=dev


python error_detection.py \
     --data_dir $DATA_DIR \
     --labels $LABELS \
     --model_name_or_path $OUTPUT_DIR \
     --output_dir $OUTPUT_DIR \
     --per_device_eval_batch_size $BATCH_SIZE \
     --seed $SEED \
     --do_pred \
     --pred_output_file ${exp}_${pred_mode}.preds.txt \
     --pred_mode $pred_mode # or test to get the test predictions

# Evaluation
paste $DATA_DIR/${pred_mode}.txt $OUTPUT_DIR/${DATASET}_${pred_mode}.preds.txt \
    > $OUTPUT_DIR/eval_data_${pred_mode}_${DATASET}.txt

python evaluate.py  --data $OUTPUT_DIR/eval_data_${pred_mode}_${DATASET}.txt \
                    --labels $LABELS \
                    --output $OUTPUT_DIR/${DATASET}_${pred_mode}.results

rm $OUTPUT_DIR/eval_data_${pred_mode}_${DATASET}.txt
```

[predictions](predictions) includes our GED models predictions and results.

