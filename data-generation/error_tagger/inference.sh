#!/usr/bin/env bash
#SBATCH -p nvidia
#SBATCH -q nlp
# use gpus
#SBATCH --gres=gpu:1
# memory
#SBATCH --mem=120GB
# Walltime format hh:mm:ss
#SBATCH --time=40:00:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err


python tagger.py \
    --train_path /scratch/ba63/gec/data/alignment/modeling_areta_tags_check/qalb14/corruption_data/train.error_tagger.json \
    --dev_path /scratch/ba63/gec/data/alignment/modeling_areta_tags_check/qalb14/corruption_data/tune.error_tagger.json \
    --do_inference \
    --embed_dim 100 \
    --hidd_dim 128 \
    --num_layers 2 \
    --batch_size 32 \
    --seed 21 \
    --do_early_stopping \
    --model_path ./model_w_morph.pt
