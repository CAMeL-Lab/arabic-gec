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


FAIRSEQ_DATA_DIR=/scratch/ba63/gec/fairseq-data/QALB-2014_new
FAIRSEQ_DIR=/home/ba63/fairseq/fairseq_cli
MODEL_DIR=/scratch/ba63/gec/models/vanilla-transformers/500_bpe
SPLIT=valid1
OUTPUT_DIR=$MODEL_DIR/outputs

fairseq-generate $FAIRSEQ_DATA_DIR/data-bin/500_joined  \
    --path  ${MODEL_DIR}/checkpoint_best.pt \
    --source-lang sent.no_ids.clean \
    --target-lang cor.no_ids \
    --gen-subset $SPLIT \
    --beam 5 --batch-size 64 --remove-bpe > $OUTPUT_DIR/generated1.txt


grep ^S $OUTPUT_DIR/generated1.txt | LC_ALL=C sort -V | cut -f2- > $OUTPUT_DIR/src1.txt
grep ^T $OUTPUT_DIR/generated1.txt | LC_ALL=C sort -V | cut -f2- > $OUTPUT_DIR/ref1.txt
grep ^H $OUTPUT_DIR/generated1.txt | LC_ALL=C sort -V | cut -f3- > $OUTPUT_DIR/hyp1.txt