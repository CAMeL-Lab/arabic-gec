#!/usr/bin/env bash
#SBATCH -p nvidia
# use gpus
#SBATCH --gres=gpu:v100:1
# memory
#SBATCH --mem=300GB
# Walltime format hh:mm:ss
#SBATCH --time=40:00:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err


MODEL=/scratch/ba63/BERT_models/AraBART
OUTPUT_DIR=/scratch/ba63/gec/models/gec/qalb14_fixes/bart_w_camelira_ged_pred_worst
TRAIN_FILE=/scratch/ba63/gec/data/bart-t5/qalb14/w_camelira/train_preds_worst.json
STEPS=500
BATCH_SIZE=32


python /home/ba63/gec/bart-t5-new/run_gec.py \
    --model_name_or_path $MODEL \
    --do_train \
    --optim adamw_torch \
    --source_lang raw \
    --target_lang cor \
    --train_file $TRAIN_FILE \
    --ged_tags /scratch/ba63/gec/data/ged/qalb14/wo_camelira/labels.txt \
    --save_steps $STEPS \
    --num_train_epochs 10 \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --max_target_length 1024 \
    --seed 42 \
    --overwrite_cache \
    --overwrite_output_dir



test_file=/scratch/ba63/gec/data/bart-t5/qalb14/w_camelira/tune_preds.json

for checkpoint in ${OUTPUT_DIR} ${OUTPUT_DIR}/checkpoint-*

do

        python /home/ba63/gec/bart-t5-new/generate.py \
                --model_name_or_path $checkpoint \
                --source_lang raw \
                --target_lang cor \
                --test_file $test_file \
                --use_ged \
                --preprocess_merges \
                --per_device_eval_batch_size 32 \
                --output_dir $checkpoint \
                --num_beams 5 \
                --num_return_sequences 1 \
                --max_target_length 1024 \
                --predict_with_generate \
                --prediction_file qalb14_tune.preds.merge_fix
done