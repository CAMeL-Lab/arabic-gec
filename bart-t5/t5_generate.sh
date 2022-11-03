#!/usr/bin/env bash
#SBATCH -p nvidia
# use gpus
#SBATCH --gres=gpu:1
# memory
#SBATCH --mem=400GB
# Walltime format hh:mm:ss
#SBATCH --time=40:00:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err

OUTPUT_DIR=/scratch/ba63/gec/models/QALB-2015/t5_lr_with_pref_binary_areta
# OUTPUT_DIR=/scratch/ba63/gec/models/t5_30
# --source_prefix "convert raw to cor: " \
# --areta_tags /scratch/ba63/gec/bart-t5-data/QALB-2014/areta.labels.binary.txt \

if [ "$1" = "all" ]; then
    for checkpoint in ${OUTPUT_DIR} ${OUTPUT_DIR}/checkpoint-*
    do
        printf "Generating outputs using: ${checkpoint}\n"
        python run_gec.py \
        --model_name_or_path $checkpoint \
        --do_predict \
        --source_lang raw \
        --target_lang cor \
        --source_prefix "convert raw to cor: " \
        --test_file /scratch/ba63/gec/bart-t5-data/ZAEBUC/dev.json \
        --per_device_eval_batch_size 16 \
        --output_dir $checkpoint \
        --num_beams 5 \
        --max_target_length 1024 \
        --overwrite_cache \
        --predict_with_generate \
        --prediction_file dev.pred.txt

    done
else
    python run_gec.py \
    --model_name_or_path $OUTPUT_DIR \
    --do_predict \
    --source_lang raw \
    --target_lang cor \
    --source_prefix "convert raw to cor: " \
    --test_file /scratch/ba63/gec/bart-t5-data/QALB-2015/dev.areta.binary.json \
    --areta_tags /scratch/ba63/gec/bart-t5-data/QALB-2015/areta.labels.binary.txt \
    --per_device_eval_batch_size 16 \
    --output_dir $OUTPUT_DIR \
    --num_beams 5 \
    --max_target_length 1024 \
    --overwrite_cache \
    --predict_with_generate \
    --prediction_file dev.pred.txt

fi
