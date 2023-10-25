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
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err




# sys=/scratch/ba63/gec/models/gec/qalb14-15/full/bart/checkpoint-11000

# sys=/scratch/ba63/gec/models/gec/qalb14/full/bart_w_camelira_ged/checkpoint-5500
# sys=/scratch/ba63/gec/models/gec/qalb14/coarse/bart_w_camelira_ged/checkpoint-5500
# sys=/scratch/ba63/gec/models/gec/qalb14/binary/bart_w_camelira_ged/checkpoint-6000

# sys=/scratch/ba63/gec/models/gec/qalb14/full/bart_w_camelira_ged/checkpoint-5500
# sys=/scratch/ba63/gec/models/gec/qalb14/coarse/bart_w_camelira_ged/checkpoint-5500
# sys=/scratch/ba63/gec/models/gec/qalb14/binary/bart_w_camelira_ged/checkpoint-6000

# sys=/scratch/ba63/gec/models/gec/qalb14-15/full/bart_w_camelira_ged/checkpoint-6000
# sys=/scratch/ba63/gec/models/gec/qalb14-15/coarse/bart_w_camelira_ged/checkpoint-6000
# sys=/scratch/ba63/gec/models/gec/qalb14-15/binary/bart_w_camelira_ged/checkpoint-6000

# sys=/scratch/ba63/gec/models/gec/mix/full/bart_w_camelira_ged/checkpoint-11000
# sys=/scratch/ba63/gec/models/gec/mix/coarse/bart_w_camelira_ged/checkpoint-8000
# sys=/scratch/ba63/gec/models/gec/mix/binary/bart_w_camelira_ged/checkpoint-11000

sys=/scratch/ba63/gec/models/gec/qalb14-15/full/bart_w_camelira
# 

test_file=/home/ba63/gec-release/data/gec/modeling/qalb15/w_camelira/full/test_L2.json
pred_file=qalb15_test-L2.preds.check

m2_edits=/home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/test/QALB-2015-L2-Test.m2
m2_edits_nopnx=/home/ba63/gec-release/data/m2edits/qalb15/qalb15_L2-test.nopnx.m2

# m2_edits=/home/ba63/gec-release/data/m2edits/zaebuc/zaebuc_dev.m2
# m2_edits_nopnx=/home/ba63/gec-release/data/m2edits/zaebuc/zaebuc_dev.nopnx.m2


printf "Running inference on ${test_file}\n"
printf "Generating outputs using: ${sys}\n"

python generate.py \
        --model_name_or_path $sys \
        --source_lang raw \
        --target_lang cor \
        --test_file $test_file \
        --m2_edits $m2_edits \
        --m2_edits_nopnx $m2_edits_nopnx \
        --per_device_eval_batch_size 16 \
        --output_dir $sys \
        --num_beams 5 \
        --num_return_sequences 1 \
        --max_target_length 1024 \
        --predict_with_generate \
        --prediction_file $pred_file


        # --use_ged \
        # --preprocess_merges \