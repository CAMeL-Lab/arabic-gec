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
OUTPUT_DIR=/scratch/ba63/gec/models/gec/${experiment}/bart_w_ged/checkpoint-3000
test_file=/scratch/ba63/gec/data/bart-t5/qalb14/wo_camelira/tune.json

# 
for checkpoint in ${OUTPUT_DIR} # ${OUTPUT_DIR}/checkpoint-*
do
        printf "Generating outputs using: ${checkpoint}\n"
        python generate.py \
            --model_name_or_path $checkpoint \
            --source_lang raw \
            --target_lang cor \
            --test_file $test_file \
            --ged_tags /scratch/ba63/gec/data/ged/qalb14/wo_camelira/labels.txt \
            --per_device_eval_batch_size 32 \
            --output_dir $checkpoint \
            --num_beams 1 \
            --max_target_length 1024 \
            --overwrite_cache \
            --prediction_file qalb14_tune.preds.check.trainer.txt

done
