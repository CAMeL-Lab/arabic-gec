#!/usr/bin/env bash
#SBATCH -p nvidia
# use gpus
#SBATCH --gres=gpu:1
# memory
#SBATCH --mem=200GB
# Walltime format hh:mm:ss
#SBATCH --time=40:00:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err


MODEL=/scratch/ba63/BERT_models/AraBART
# MODEL=/scratch/ba63/BERT_models/AraT5-base
# MODEL=/scratch/ba63/BERT_models/AraT5-msa-base
# --source_prefix "convert raw to cor: " \
# --validation_file /scratch/ba63/gec/bart-t5-data/ZAEBUC/dev.json

OUTPUT_DIR=/scratch/ba63/gec/models/ZAEBUC/bart_with_binary_areta
STEPS=500 #1500 for MIX
BATCH_SIZE=32 #16 for MIX

python run_gec.py \
    --model_name_or_path $MODEL \
    --do_train \
    --source_lang raw \
    --target_lang cor \
    --train_file /scratch/ba63/gec/bart-t5-data/ZAEBUC/train.areta.binary.json \
    --areta_tags  /scratch/ba63/gec/bart-t5-data/ZAEBUC/areta.labels.binary.txt \
    --save_steps $STEPS \
    --num_train_epochs 10 \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --max_target_length 1024 \
    --seed 42 \
    --overwrite_cache \
    --overwrite_output_dir
