#!/usr/bin/env bash
#SBATCH -p nvidia
#SBATCH -q nlp
# use gpus
#SBATCH --gres=gpu:v100:1
# memory
# SBATCH --mem=120GB
# Walltime format hh:mm:ss
#SBATCH --time=47:59:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err


MODEL=/scratch/ba63/BERT_models/AraBART

# OUTPUT_DIR=/scratch/ba63/gec-release/models/gec/qalb14/full/bart_w_camelira
# TRAIN_FILE=/home/ba63/gec-release/data/gec/modeling/qalb14/w_camelira/full/train.json

# OUTPUT_DIR=/scratch/ba63/gec-release/models/gec/qalb14-15/full/bart_w_camelira
# TRAIN_FILE=/home/ba63/gec-release/data/gec/modeling/qalb14-15/w_camelira/full/train.json

OUTPUT_DIR=/scratch/ba63/gec-release/models/gec/mix/full/bart_w_camelira
TRAIN_FILE=/home/ba63/gec-release/data/gec/modeling/mix/w_camelira/full/train.json



STEPS=1000 # 500 for qalb14 and 1000 for mix
BATCH_SIZE=16 # 32 for qalb14 and 16 for mix


python /home/ba63/gec-release/gec/run_gec.py \
    --model_name_or_path $MODEL \
    --do_train \
    --optim adamw_torch \
    --source_lang raw \
    --target_lang cor \
    --train_file $TRAIN_FILE \
    --save_steps $STEPS \
    --num_train_epochs 10 \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --max_target_length 1024 \
    --seed 42 \
    --overwrite_cache \
    --overwrite_output_dir
