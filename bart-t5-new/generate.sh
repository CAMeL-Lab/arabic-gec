#!/usr/bin/env bash
#SBATCH -p nvidia
#SBATCH --reservation=nlp-gpu
# use gpus
#SBATCH --gres=gpu:v100:1
# memory
#SBATCH --mem=200GB
# Walltime format hh:mm:ss
#SBATCH --time=40:00:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err


sys=/scratch/ba63/gec/models/gec/mix/bart_w_ged_pred_worst/checkpoint-5000
test_file=/scratch/ba63/gec/data/bart-t5/qalb15/wo_camelira/dev.json
pred_file=qalb15_dev.preds


for checkpoint in $sys #$sys/checkpoint-*
do
        printf "Running inference on ${test_file}\n"
        printf "Generating outputs using: ${checkpoint}\n"

        python generate.py \
                --model_name_or_path $checkpoint \
                --source_lang raw \
                --target_lang cor \
                --test_file $test_file \
                --use_ged \
                --preprocess_merges \
                --per_device_eval_batch_size 16 \
                --output_dir $checkpoint \
                --num_beams 5 \
                --num_return_sequences 1 \
                --max_target_length 1024 \
                --predict_with_generate \
                --prediction_file $pred_file

done