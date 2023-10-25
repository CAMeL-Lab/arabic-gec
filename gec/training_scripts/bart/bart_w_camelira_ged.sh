#!/usr/bin/env bash
#SBATCH -p nvidia
# SBATCH -q nlp
# SBATCH --reservation=v100_nlp
# use gpus
#SBATCH --gres=gpu:v100:1
# memory
# SBATCH --mem=120GB
# Walltime format hh:mm:ss
#SBATCH --time=40:00:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err


MODEL=/scratch/ba63/BERT_models/AraBART

# OUTPUT_DIR=/scratch/ba63/gec/models/gec/qalb14/full/bart_w_camelira_ged_check
# TRAIN_FILE=/home/ba63/gec-release/data/gec/modeling/qalb14/w_camelira/full/train.oracle.json
# LABELS=/home/ba63/gec-release/data/ged/qalb14/w_camelira/full/labels.txt

# OUTPUT_DIR=/scratch/ba63/gec/models/gec/qalb14/coarse/bart_w_camelira_ged_check
# TRAIN_FILE=/home/ba63/gec-release/data/gec/modeling/qalb14/w_camelira/coarse/train.oracle.json
# LABELS=/home/ba63/gec-release/data/ged/qalb14/w_camelira/coarse/labels.txt

# OUTPUT_DIR=/scratch/ba63/gec/models/gec/qalb14/binary/bart_w_camelira_ged_check
# TRAIN_FILE=/home/ba63/gec-release/data/gec/modeling/qalb14/w_camelira/binary/train.oracle.json
# LABELS=/home/ba63/gec-release/data/ged/qalb14/w_camelira/binary/labels.txt


# OUTPUT_DIR=/scratch/ba63/gec/models/gec/qalb14-15/full/bart_w_camelira_ged_check
# TRAIN_FILE=/home/ba63/gec-release/data/gec/modeling/qalb14-15/w_camelira/full/train.oracle.json
# LABELS=/home/ba63/gec-release/data/ged/qalb14-15/w_camelira/full/labels.txt

# OUTPUT_DIR=/scratch/ba63/gec/models/gec/qalb14-15/coarse/bart_w_camelira_ged_check
# TRAIN_FILE=/home/ba63/gec-release/data/gec/modeling/qalb14-15/w_camelira/coarse/train.oracle.json
# LABELS=/home/ba63/gec-release/data/ged/qalb14-15/w_camelira/coarse/labels.txt

# OUTPUT_DIR=/scratch/ba63/gec/models/gec/qalb14-15/binary/bart_w_camelira_ged_check
# TRAIN_FILE=/home/ba63/gec-release/data/gec/modeling/qalb14-15/w_camelira/binary/train.oracle.json
# LABELS=/home/ba63/gec-release/data/ged/qalb14-15/w_camelira/binary/labels.txt


# OUTPUT_DIR=/scratch/ba63/gec/models/gec/mix/full/bart_w_camelira_ged_check
# TRAIN_FILE=/home/ba63/gec-release/data/gec/modeling/mix/w_camelira/full/train.oracle.json
# LABELS=/home/ba63/gec-release/data/ged/mix/w_camelira/full/labels.txt

# OUTPUT_DIR=/scratch/ba63/gec/models/gec/mix/coarse/bart_w_camelira_ged_check
# TRAIN_FILE=/home/ba63/gec-release/data/gec/modeling/mix/w_camelira/coarse/train.oracle.json
# LABELS=/home/ba63/gec-release/data/ged/mix/w_camelira/coarse/labels.txt

OUTPUT_DIR=/scratch/ba63/gec/models/gec/mix/binary/bart_w_camelira_ged_check
TRAIN_FILE=/home/ba63/gec-release/data/gec/modeling/mix/w_camelira/binary/train.oracle.json
LABELS=/home/ba63/gec-release/data/ged/mix/w_camelira/binary/labels.txt

STEPS=1000 # 500 for qalb14 and 1000 for mix
BATCH_SIZE=16 # 32 for qalb14 and 16 for mix


python /home/ba63/gec-release/gec/run_gec.py \
    --model_name_or_path $MODEL \
    --do_train \
    --optim adamw_torch \
    --source_lang raw \
    --target_lang cor \
    --train_file $TRAIN_FILE \
    --ged_tags  $LABELS \
    --save_steps $STEPS \
    --num_train_epochs 10 \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --max_target_length 1024 \
    --seed 42 \
    --overwrite_cache \
    --overwrite_output_dir
