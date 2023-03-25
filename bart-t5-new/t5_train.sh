#!/usr/bin/env bash
#SBATCH -p nvidia
# SBATCH -q nlp
# use gpus
#SBATCH --gres=gpu:1
# memory
#SBATCH --mem=120GB
# Walltime format hh:mm:ss
#SBATCH --time=40:00:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err


MODEL=/scratch/ba63/BERT_models/AraT5-base
OUTPUT_DIR=/scratch/ba63/gec/models/gec/qalb14/t5_w_camelira_ged
STEPS=1500 #5000 for MIX / 1500 for qalb14/qalb15/zaebuc
BATCH_SIZE=16 #8 for MIX/qalb15/zaebuc / 16 for qalb14


python run_gec.py \
    --model_name_or_path $MODEL \
    --do_train \
    --optim adamw_torch \
    --source_lang raw \
    --target_lang cor \
    --ged_tags /scratch/ba63/gec/data/ged/qalb14/w_camelira/labels.txt \
    --train_file  /scratch/ba63/gec/data/bart-t5/qalb14/w_camelira/train.json \
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
