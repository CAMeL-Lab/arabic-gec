#!/usr/bin/env bash
# SBATCH --reservation=nlp
#SBATCH -p nvidia
# SBATCH -q nlp
# use gpus
#SBATCH --gres=gpu:1
# memory
#SBATCH --mem=200GB
# Walltime format hh:mm:ss
#SBATCH --time=40:00:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err


MODEL=/scratch/ba63/BERT_models/AraT5-base
# MODEL=/scratch/ba63/BERT_models/AraT5-msa-base
# --source_prefix "convert raw to cor: " \
# --validation_file /scratch/ba63/gec/bart-t5-data/ZAEBUC/dev.json

OUTPUT_DIR=/scratch/ba63/gec/models/MIX/t5_lr_with_pref_full_areta_30_prof_vec
STEPS=5000 #5000 for MIX / 1500 for default
BATCH_SIZE=8 #8 for MIX / 16 for default

python run_gec_dev_one_hot.py \
    --model_name_or_path $MODEL \
    --do_train \
    --source_lang raw \
    --target_lang cor \
    --source_prefix "convert raw to cor: " \
    --train_file /scratch/ba63/gec/bart-t5-data/MIX/train-prof.areta.json \
    --areta_tags /scratch/ba63/gec/bart-t5-data/MIX/areta.29.base.txt \
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
