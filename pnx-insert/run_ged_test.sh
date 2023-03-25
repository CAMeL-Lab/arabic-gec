#!/bin/bash
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

#################################
# ERROR DETECTION TEST EVAL SCRIPT
#################################

export DATA_DIR=/scratch/ba63/gec/data/alignment/modeling_areta_tags/qalb14/modeling_ged
export MAX_LENGTH=256
export OUTPUT_DIR=/scratch/ba63/gec/data/alignment/modeling_areta_tags/qalb14/modeling_ged/model_new_latest_fix_reg
export BATCH_SIZE=32
export SEED=42


python error_detection.py \
--data_dir $DATA_DIR \
--labels $DATA_DIR/labels.txt \
--model_name_or_path $OUTPUT_DIR \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--per_device_eval_batch_size $BATCH_SIZE \
--seed $SEED \
--do_pred \
--pred_output_file tune_qalb14_predictions.txt \
--pred_mode tune # or dev or test to get the dev or test predictions
