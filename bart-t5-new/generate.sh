#!/usr/bin/env bash
#SBATCH -p nvidia
# SBATCH -q nlp
# use gpus
#SBATCH --gres=gpu:1
# memory
#SBATCH --mem=120GB
# Walltime format hh:mm:ss
#SBATCH --time=40:00:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err


experiment=qalb14
OUTPUT_DIR=/scratch/ba63/gec/models/gec/qalb14/t5_w_ged_pred_worst
test_file=/scratch/ba63/gec/data/bart-t5/qalb14/wo_camelira/tune.json
# 

printf "Running inference on ${test_file}\n"
for checkpoint in ${OUTPUT_DIR} ${OUTPUT_DIR}/checkpoint-*
do
        printf "Generating outputs using: ${checkpoint}\n"
        python generate.py \
                --model_name_or_path $checkpoint \
                --source_lang raw \
                --target_lang cor \
                --test_file $test_file \
                --ged_tags /scratch/ba63/gec/data/ged/qalb14/wo_camelira/labels.txt \
                --per_device_eval_batch_size 16 \
                --output_dir $checkpoint \
                --num_beams 5 \
                --num_return_sequences 5 \
                --max_target_length 1024 \
                --overwrite_cache \
                --predict_with_generate \
                --prediction_file qalb14_tune.preds.check

done
