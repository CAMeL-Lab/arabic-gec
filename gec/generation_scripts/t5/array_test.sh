#!/usr/bin/env bash
#SBATCH -p nvidia
# SBATCH -q nlp
# use gpus
#SBATCH --gres=gpu:v100:1
# memory
# SBATCH --mem=200GB
# Walltime format hh:mm:ss
#SBATCH --time=40:00:00
# Output and error files
#SBATCH -o job.%J.%A.out
#SBATCH -e job.%J.%A.err
#SBATCH --array=0-3

sys=/scratch/ba63/gec/models/gec/mix/binary/t5_w_camelira_ged
pred_file=zaebuc_dev.preds.oraclee
test_file=/scratch/ba63/gec/data/bart-t5/zaebuc/binary/w_camelira/dev.json
m2_edits=/scratch/ba63/gec/data/alignment/m2_files/zaebuc_dev.m2
m2_edits_nopnx=/scratch/ba63/gec/data/alignment/m2_files/zaebuc_dev.nopnx.m2

# Array of checkpoints
checkpoints=(
  "${sys}/checkpoint-66000"
  "${sys}/checkpoint-69000"
  "${sys}/checkpoint-72000"
  "${sys}/checkpoint-9000"
)

checkpoint=${checkpoints[$SLURM_ARRAY_TASK_ID]}

nvidia-smi
python /home/ba63/gec/bart-t5-new/generate.py \
    --model_name_or_path "$checkpoint" \
    --source_lang raw \
    --target_lang cor \
    --use_ged \
    --preprocess_merges \
    --test_file "$test_file" \
    --m2_edits "$m2_edits" \
    --m2_edits_nopnx "$m2_edits_nopnx" \
    --per_device_eval_batch_size 16 \
    --output_dir "$checkpoint" \
    --num_beams 5 \
    --num_return_sequences 1 \
    --max_target_length 1024 \
    --predict_with_generate \
    --prediction_file "$pred_file"
