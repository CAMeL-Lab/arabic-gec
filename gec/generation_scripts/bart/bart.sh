#!/usr/bin/env bash
#SBATCH -p nvidia
# use gpus
#SBATCH --gres=gpu:v100:1
# memory
# SBATCH --mem=100GB
# Walltime format hh:mm:ss
#SBATCH --time=40:00:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err


# /scratch/ba63/gec/models/gec/qalb14-15/full/bart/checkpoint-11000
# /scratch/ba63/gec/models/gec/mix/full/bart/checkpoint-8000


sys=/scratch/ba63/gec/models/gec/qalb14/full/bart/checkpoint-3000
pred_file=qalb14_test.preds.check
test_file=/home/ba63/gec-release/data/gec/modeling/qalb14/wo_camelira/full/test.json
m2_edits=/scratch/ba63/gec/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/test/QALB-2014-L1-Test.m2
m2_edits_nopnx=/home/ba63/gec-release/data/m2edits/qalb14/qalb14_test.nopnx.m2


# sys=/scratch/ba63/gec/models/gec/qalb14-15/full/bart/checkpoint-11000
# pred_file=qalb15_test-L2.preds.txt.check
# test_file=/home/ba63/gec-release/data/gec/modeling/qalb15/wo_camelira/full/test.json
# m2_edits=/scratch/ba63/gec/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/test/QALB-2015-L2-Test.m2
# m2_edits_nopnx=/home/ba63/gec-release/data/m2edits/qalb15/qalb15_test.nopnx.m2


sys=/scratch/ba63/gec/models/gec/mix/full/bart/checkpoint-8000
pred_file=zaebuc_test.preds.check
test_file=/home/ba63/gec-release/data/gec/modeling/zaebuc/wo_camelira/full/test.json
m2_edits=/home/ba63/gec-release/data/m2edits/zabuc/zaebuc_test.m2
m2_edits_nopnx=/home/ba63/gec-release/data/m2edits/zabuc/zaebuc_test.nopnx.m2


nvidia-smi

for checkpoint in ${sys} ${sys}/checkpoint-*
do
        python /home/ba63/gec-release/gec/generate.py \
                --model_name_or_path $checkpoint \
                --source_lang raw \
                --target_lang cor \
                --test_file $test_file \
                --m2_edits $m2_edits \
                --m2_edits_nopnx $m2_edits_nopnx \
                --per_device_eval_batch_size 8 \
                --output_dir $checkpoint \
                --num_beams 5 \
                --num_return_sequences 1 \
                --max_target_length 1024 \
                --predict_with_generate \
                --prediction_file $pred_file
done

