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


FAIRSEQ_DATA_DIR=/scratch/ba63/gec/fairseq-data/qalb14
FAIRSEQ_DIR=/home/ba63/fairseq/fairseq_cli

CUDA_VISIBLE_DEVICES=0 python -u $FAIRSEQ_DIR/train.py \
    $FAIRSEQ_DATA_DIR/data-bin/50000_joined/ \
    --source-lang sent.no_ids.clean.dediac \
    --target-lang cor.no_ids.dediac \
    --valid-subset valid1 \
    --ignore-unused-valid-subsets \
    --log-format simple \
    --max-epoch 50 \
    --arch transformer \
    --max-tokens 4096 \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --lr 0.0005 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --warmup-init-lr 1e-07 \
    --stop-min-lr 1e-09 \
    --dropout 0.3 \
    --clip-norm 1.0 \
    --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --no-epoch-checkpoints \
    --share-all-embeddings \
    --seed 42 \
    --save-dir /scratch/ba63/gec/models/qalb14/vanilla-transformers/50k_bpe
