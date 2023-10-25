#!/usr/bin/env bash
#SBATCH -p nvidia
# SBATCH -q nlp
#SBATCH --reservation=v100_nlp
# use gpus
#SBATCH --gres=gpu:v100:1
# memory
# SBATCH --mem=200GB
# Walltime format hh:mm:ss
#SBATCH --time=40:00:00
# Output and error files
#SBATCH -o job.%J.%A.out
#SBATCH -e job.%J.%A.err
#SBATCH --array=0-12

sys=/scratch/ba63/gec/models/gec/mix/binary/bart_w_camelira_ged
pred_file=zaebuc_dev.preds.oracle.check
test_file=/home/ba63/gec-release/data/gec/modeling/zaebuc/w_camelira/binary/dev.oracle.json

# m2_edits=/home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/dev/QALB-2015-L2-Dev.m2
# m2_edits_nopnx=/home/ba63/gec-release/data/m2edits/qalb15/qalb15_dev.nopnx.m2

# m2_edits=/home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/dev/QALB-2015-L2-Dev.m2
# m2_edits_nopnx=/home/ba63/gec-release/data/m2edits/qalb15/qalb15_dev.nopnx.m2

m2_edits=/home/ba63/gec-release/data/m2edits/zaebuc/zaebuc_dev.m2
m2_edits_nopnx=/home/ba63/gec-release/data/m2edits/zaebuc/zaebuc_dev.nopnx.m2


# Array of checkpoints
checkpoints=(
  "${sys}"
  "${sys}/checkpoint-1000"
  "${sys}/checkpoint-10000"
  "${sys}/checkpoint-11000"
  "${sys}/checkpoint-12000"
  "${sys}/checkpoint-2000"
  "${sys}/checkpoint-3000"
  "${sys}/checkpoint-4000"
  "${sys}/checkpoint-5000"
  "${sys}/checkpoint-6000"
  "${sys}/checkpoint-7000"
  "${sys}/checkpoint-8000"
  "${sys}/checkpoint-9000"
)


# checkpoints=(
#   "${sys}"
#   "${sys}/checkpoint-1000"
#   "${sys}/checkpoint-1500"
#   "${sys}/checkpoint-2000"
#   "${sys}/checkpoint-2500"
#   "${sys}/checkpoint-3000"
#   "${sys}/checkpoint-3500"
#   "${sys}/checkpoint-4000"
#   "${sys}/checkpoint-4500"
#   "${sys}/checkpoint-500"
#   "${sys}/checkpoint-5000"
#   "${sys}/checkpoint-5500"
#   "${sys}/checkpoint-6000"
# )



checkpoint=${checkpoints[$SLURM_ARRAY_TASK_ID]}

python /home/ba63/gec-release/gec/generate.py \
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


