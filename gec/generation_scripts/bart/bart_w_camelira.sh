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



# sys=/scratch/ba63/gec/models/gec/qalb14/full/bart_w_camelira
# pred_file=qalb14_dev.preds
# test_file=/scratch/ba63/gec/data/bart-t5/qalb14/full/w_camelira/dev_preds.json
# m2_edits=/scratch/ba63/gec/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/dev/QALB-2014-L1-Dev.m2
# m2_edits_nopnx=/scratch/ba63/gec/data/alignment/m2_files/qalb14_dev.nopnx.m2


# sys=/scratch/ba63/gec/models/gec/qalb14-15/full/bart_w_camelira
# pred_file=qalb15_dev.preds
# test_file=/scratch/ba63/gec/data/bart-t5/qalb15/full/w_camelira/dev_qalb14-15_preds.json
# m2_edits=/scratch/ba63/gec/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/dev/QALB-2015-L2-Dev.m2
# m2_edits_nopnx=/scratch/ba63/gec/data/alignment/m2_files/qalb15_dev.nopnx.m2

sys=/scratch/ba63/gec/models/gec/mix/full/bart_w_camelira
pred_file=zaebuc_dev.preds
test_file=/scratch/ba63/gec/data/bart-t5/zaebuc/full/w_camelira/dev_mix_preds.json
m2_edits=/scratch/ba63/gec/data/alignment/m2_files/zaebuc_dev.m2
m2_edits_nopnx=/scratch/ba63/gec/data/alignment/m2_files/zaebuc_dev.nopnx.m2


for checkpoint in ${sys} ${sys}/checkpoint-*
do
        python /home/ba63/gec/bart-t5-new/generate.py \
                --model_name_or_path $checkpoint \
                --source_lang raw \
                --target_lang cor \
                --test_file $test_file \
                --m2_edits $m2_edits \
                --m2_edits_nopnx $m2_edits_nopnx \
                --per_device_eval_batch_size 16 \
                --output_dir $checkpoint \
                --num_beams 5 \
                --num_return_sequences 1 \
                --max_target_length 1024 \
                --predict_with_generate \
                --prediction_file $pred_file
done

