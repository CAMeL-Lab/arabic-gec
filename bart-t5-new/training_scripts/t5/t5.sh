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

nvidia-smi

MODEL=/scratch/ba63/BERT_models/AraT5-base
OUTPUT_DIR=/scratch/ba63/gec/models/gec/qalb14_fixes/t5
TRAIN_FILE=/scratch/ba63/gec/data/bart-t5/qalb14/wo_camelira/train.json
STEPS=1500
BATCH_SIZE=16


python /home/ba63/gec/bart-t5-new/run_gec.py \
    --model_name_or_path $MODEL \
    --do_train \
    --optim adamw_torch \
    --source_lang raw \
    --target_lang cor \
    --train_file  $TRAIN_FILE \
    --save_steps $STEPS \
    --num_train_epochs 30 \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --max_target_length 1024 \
    --seed 42 \
    --learning_rate 1e-04 \
    --overwrite_cache \
    --overwrite_output_dir



test_file=/scratch/ba63/gec/data/bart-t5/qalb14/wo_camelira/tune_preds.json

for checkpoint in ${OUTPUT_DIR} ${OUTPUT_DIR}/checkpoint-*

do

        python /home/ba63/gec/bart-t5-new/generate.py \
                --model_name_or_path $checkpoint \
                --source_lang raw \
                --target_lang cor \
                --test_file $test_file \
                --per_device_eval_batch_size 16 \
                --output_dir $checkpoint \
                --num_beams 5 \
                --num_return_sequences 1 \
                --max_target_length 1024 \
                --predict_with_generate \
                --prediction_file qalb14_tune.preds
done


