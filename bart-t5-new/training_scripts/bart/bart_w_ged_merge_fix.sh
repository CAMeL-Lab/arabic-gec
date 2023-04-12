#!/usr/bin/env bash
#SBATCH -p nvidia
# use gpus
#SBATCH --gres=gpu:v100:1
# memory
#SBATCH --mem=200GB
# Walltime format hh:mm:ss
#SBATCH --time=40:00:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err


MODEL=/scratch/ba63/BERT_models/AraBART
OUTPUT_DIR=/scratch/ba63/gec/models/gec/qalb14/bart_w_ged_merge_fix
TRAIN_FILE=/scratch/ba63/gec/data/bart-t5/qalb14/wo_camelira/train.json
LABELS=/scratch/ba63/gec/data/ged++/qalb14/wo_camelira/labels.txt

# OUTPUT_DIR=/scratch/ba63/gec/models/gec/mix/bart_w_ged_merge_fix
# TRAIN_FILE=/scratch/ba63/gec/data/bart-t5/mix/wo_camelira/train.json
# LABELS=/scratch/ba63/gec/data/ged++/mix/wo_camelira/labels.txt

STEPS=500 # 500 for qalb14 and 1000 for mix
BATCH_SIZE=32 # 32 for qalb14 and 16 for mix



python /home/ba63/gec/bart-t5-new/run_gec.py \
    --model_name_or_path $MODEL \
    --do_train \
    --optim adamw_torch \
    --source_lang raw \
    --target_lang cor \
    --train_file $TRAIN_FILE \
    --ged_tags $LABELS \
    --preprocess_merges \
    --save_steps $STEPS \
    --num_train_epochs 10 \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --max_target_length 1024 \
    --seed 42 \
    --overwrite_cache \
    --overwrite_output_dir


# test_file=/scratch/ba63/gec/data/bart-t5/qalb14/wo_camelira/tune_preds.json
# test_file=/scratch/ba63/gec/data/bart-t5/mix/wo_camelira/tune_preds.json
# PRED_FILE=mix_tune.preds.merge_fix
# for checkpoint in ${OUTPUT_DIR} ${OUTPUT_DIR}/checkpoint-*

# do

#         python /home/ba63/gec/bart-t5-new/generate.py \
#                 --model_name_or_path $checkpoint \
#                 --source_lang raw \
#                 --target_lang cor \
#                 --test_file $test_file \
#                 --use_ged \
#                 --preprocess_merges \
#                 --per_device_eval_batch_size 32 \
#                 --output_dir $checkpoint \
#                 --num_beams 5 \
#                 --num_return_sequences 1 \
#                 --max_target_length 1024 \
#                 --predict_with_generate \
#                 --prediction_file $PRED_FILE
# done