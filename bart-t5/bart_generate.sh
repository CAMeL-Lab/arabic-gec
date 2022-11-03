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

OUTPUT_DIR=/scratch/ba63/gec/models/ZAEBUC/bart_with_binary_areta
# OUTPUT_DIR=/scratch/ba63/gec/models/t5_30
# --source_prefix "convert raw to cor: " \

# OUTPUT_DIR=/scratch/ba63/gec/models/bart_30
# OUTPUT_DIR=/scratch/ba63/gec/models/MIX/bart

if [ "$1" = "all" ]; then
    for checkpoint in ${OUTPUT_DIR} ${OUTPUT_DIR}/checkpoint-*
    do
        printf "Generating outputs using: ${checkpoint}\n"
        python run_gec.py \
        --model_name_or_path $checkpoint \
        --do_predict \
        --source_lang raw \
        --target_lang cor \
        --test_file /scratch/ba63/gec/bart-t5-data/QALB-2014/tune.json \
        --per_device_eval_batch_size 32 \
        --output_dir $checkpoint \
        --num_beams 5 \
        --max_target_length 1024 \
        --overwrite_cache \
        --predict_with_generate \
        --prediction_file tune.qalb14.pred.txt

    done
else
    python run_gec.py \
    --model_name_or_path $OUTPUT_DIR \
    --do_predict \
    --source_lang raw \
    --target_lang cor \
    --test_file /scratch/ba63/gec/bart-t5-data/ZAEBUC/dev.areta.binary.json \
    --areta_tags /scratch/ba63/gec/bart-t5-data/ZAEBUC/areta.labels.binary.txt \
    --per_device_eval_batch_size 64 \
    --output_dir $OUTPUT_DIR \
    --num_beams 5 \
    --max_target_length 1024 \
    --overwrite_cache \
    --predict_with_generate \
    --prediction_file dev.pred.txt

fi
