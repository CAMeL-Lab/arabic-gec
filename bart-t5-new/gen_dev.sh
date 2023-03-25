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
OUTPUT_DIR=/scratch/ba63/gec/models/gec/${experiment}/bart_w_ged_pred_worst_xattn
test_file=/scratch/ba63/gec/data/bart-t5/qalb14/wo_camelira/tune_preds.json



python run_gec_dev.py \
    --model_name_or_path $OUTPUT_DIR \
    --do_predict \
    --source_lang raw \
    --target_lang cor \
    --ged_tags /scratch/ba63/gec/data/ged/qalb14/wo_camelira/labels.txt \
    --test_file $test_file \
    --per_device_eval_batch_size 32 \
    --output_dir $OUTPUT_DIR \
    --num_beams 5 \
    --max_target_length 1024 \
    --overwrite_cache \
    --predict_with_generate \
    --prediction_file qalb14_tune.preds.no_oracle.txt
