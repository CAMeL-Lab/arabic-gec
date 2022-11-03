#!/usr/bin/env bash
#SBATCH -p nvidia
# use gpus
#SBATCH --gres=gpu:1
# memory
#SBATCH --mem=150GB
# Walltime format hh:mm:ss
#SBATCH --time=40:00:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err


FAIRSEQ_DATA_DIR=/scratch/ba63/gec/fairseq-data/QALB-2014

CUDA_VISIBLE_DEVICES=0 fairseq-train \
    $FAIRSEQ_DATA_DIR/data-bin/500_joined/ \
    --source-lang sent.no_ids.clean \
    --target-lang cor.no_ids \
    --arch transformer \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr 5e-4 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --dropout 0.3 \
    --weight-decay 0.0001 \
    --share-all-embeddings \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-tokens 4096 \
    --max-epoch 50 \
    --best-checkpoint-metric loss \
    --seed 42 \
    --save-dir /scratch/ba63/gec/models/vanilla-transformers-old/500_bpe \
