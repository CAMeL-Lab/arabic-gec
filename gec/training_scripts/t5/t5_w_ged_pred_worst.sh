#!/usr/bin/env bash
#SBATCH -p nvidia
# use gpus
#SBATCH --gres=gpu:v100:1
# memory
#SBATCH --mem=225GB
# Walltime format hh:mm:ss
#SBATCH --time=40:00:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err


MODEL=/scratch/ba63/BERT_models/AraT5-base

# OUTPUT_DIR=/scratch/ba63/gec/models/gec/qalb14/binary/t5_w_ged_pred_worst_check
# TRAIN_FILE=/scratch/ba63/gec/data/bart-t5/qalb14/binary/wo_camelira/train_preds_worst.json
# LABELS=/scratch/ba63/gec/data/ged++/qalb14/binary/wo_camelira/labels.txt


# OUTPUT_DIR=/scratch/ba63/gec/models/gec/qalb14-15/binary/t5_w_ged_pred_worst_check
# TRAIN_FILE=/scratch/ba63/gec/data/bart-t5/qalb14-15/binary/wo_camelira/train_preds_worst.json
# LABELS=/scratch/ba63/gec/data/ged++/qalb14-15/binary/wo_camelira/labels.txt


OUTPUT_DIR=/scratch/ba63/gec/models/gec/mix/binary/t5_w_ged_pred_worst_check
TRAIN_FILE=/scratch/ba63/gec/data/bart-t5/mix/binary/wo_camelira/train_preds_worst.json
LABELS=/scratch/ba63/gec/data/ged++/mix/binary/wo_camelira/labels.txt


STEPS=3000 # 1500 for qalb14 3000 for mix
BATCH_SIZE=8 # 16 for qalb14 8 for mix


python /home/ba63/gec-release/gec/run_gec.py \
    --model_name_or_path $MODEL \
    --do_train \
    --optim adamw_torch \
    --source_lang raw \
    --target_lang cor \
    --train_file  $TRAIN_FILE \
    --ged_tags $LABELS \
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