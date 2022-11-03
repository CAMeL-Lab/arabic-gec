#!/usr/bin/env bash
#SBATCH -p nvidia
#SBATCH -q nlp
# use gpus
#SBATCH --gres=gpu:1
# memory
#SBATCH --mem=150GB
# Walltime format hh:mm:ss
#SBATCH --time=40:00:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err


FAIRSEQ_DATA_DIR=/scratch/ba63/gec/fairseq-data/QALB-2014_new/bpe-500/
FAIRSEQ_DIR=/home/ba63/fairseq/fairseq_cli
MODEL_DIR=/scratch/ba63/gec/models/vanilla-transformers
beam=5

python -u $FAIRSEQ_DIR/interactive.py \
  --input /scratch/ba63/gec/fairseq-data/QALB-2014_new/bpe-500/tune.sent.no_ids.clean \
  --path $MODEL_DIR/checkpoint_best.pt \
  --beam $beam \
  --no-progress-bar \
  --buffer-size 1024 \
  --batch-size 32 \
  --log-format simple \
  --remove-bpe
  

# < $FAIRSEQ_DATA_DIR/bpe-20k/dev.sent.no_ids.clean  > ./generated.nbest.tok

# python -u /home/ba63/fairseq/fairseq_cli/interactive.py /scratch/ba63/gec/fairseq-data/QALB-2014/data-bin/5k_joined \
#   --input /scratch/ba63/gec/fairseq-data/QALB-2014/bpe-5k/dev.sent.no_ids.clean \
#   --path /scratch/ba63/gec/models/vanilla-transformers/5k_bpe/checkpoint_best.pt \
#   --source-lang sent.no_ids.clean \
#   --target-lang cor.no_ids \
#   --beam 5 \
#   --no-progress-bar \
#   --buffer-size 1024 \
#   --batch-size 32 \
#   --log-format simple \
#   --remove-bpe
  