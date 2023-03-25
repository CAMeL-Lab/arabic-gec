#!/bin/bash
#SBATCH -q nlp
#SBATCH -p nvidia
# use gpus
#SBATCH --gres=gpu:1
# memory
#SBATCH --mem=120000
# Walltime format hh:mm:ss
#SBATCH --time=47:59:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err

nvidia-smi
module purge


####################################
# ERROR DETECTION FINE-TUNING SCRIPT
####################################
export DATA_DIR=/scratch/ba63/gec/data/ged/qalb14/wo_camelira_binary
export MAX_LENGTH=256 # 256 for QALB-2014 and 512 for QALB-2015 and ZAEBUC and MIX
export BERT_MODEL=/scratch/ba63/BERT_models/bert-base-arabic-camelbert-msa
# export BERT_MODEL=/scratch/ba63/BERT_models/AraBART
export OUTPUT_DIR=/scratch/ba63/gec/models/ged/qalb14/wo_camelira_binary
export BATCH_SIZE=32
export NUM_EPOCHS=10 # 10
export SAVE_STEPS=500
export SEED=42

# --add_class_weights

python error_detection.py \
    --data_dir $DATA_DIR \
    --labels $DATA_DIR/labels.txt \
    --model_name_or_path $BERT_MODEL \
    --output_dir $OUTPUT_DIR \
    --max_seq_length  $MAX_LENGTH \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --save_steps $SAVE_STEPS \
    --seed $SEED \
    --do_train \
    --overwrite_output_dir
