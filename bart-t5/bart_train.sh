#!/usr/bin/env bash
# SBATCH --reservation=nlp
#SBATCH -p nvidia
# SBATCH -q nlp
# use gpus
#SBATCH --gres=gpu:1
# memory
#SBATCH --mem=300GB
# Walltime format hh:mm:ss
#SBATCH --time=40:00:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err


MODEL=/scratch/ba63/BERT_models/AraBART

OUTPUT_DIR=/scratch/ba63/gec/models/MIX/bart_with_full_areta_encoder_50
STEPS=500 #1500 for MIX / 500 default
BATCH_SIZE=32 #16 for MIX / 32 default

python run_gec_dev.py \
    --model_name_or_path $MODEL \
    --do_train \
    --source_lang raw \
    --target_lang cor \
    --train_file /scratch/ba63/gec/bart-t5-data/MIX/train.areta.json \
    --areta_tags /scratch/ba63/gec/bart-t5-data/MIX/areta.labels.txt \
    --save_steps $STEPS \
    --num_train_epochs 50 \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --max_target_length 1024 \
    --seed 42 \
    --overwrite_cache \
    --overwrite_output_dir
