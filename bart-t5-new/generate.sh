#!/usr/bin/env bash
#SBATCH -p nvidia
# SBATCH -q nlp
# use gpus
#SBATCH --gres=gpu:1
# memory
#SBATCH --mem=150GB
# Walltime format hh:mm:ss
#SBATCH --time=40:00:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err


sys=/scratch/ba63/gec/models/gec/qalb14_updated/bart
test_file=/scratch/ba63/gec/data/bart-t5/qalb14/wo_camelira/tune_preds.json


printf "Running inference on ${test_file}\n"

rm $sys*/qalb*
rm $sys*/checkpoint*/qalb*

rm $sys*/m2*
rm $sys*/checkpoint*/m2*

# for checkpoint in ${sys} # ${sys}/checkpoint-*

# do
#         printf "Generating outputs using: ${checkpoint}\n"
#                 python generate.py \
#                 --model_name_or_path $checkpoint \
#                 --source_lang raw \
#                 --target_lang cor \
#                 --use_ged_tags \
#                 --preprocess_merges \
#                 --test_file $test_file \
#                 --per_device_eval_batch_size 64 \
#                 --output_dir $checkpoint \
#                 --num_beams 5 \
#                 --num_return_sequences 1 \
#                 --max_target_length 1024 \
#                 --predict_with_generate \
#                 --prediction_file qalb14_tune.preds.check
# done