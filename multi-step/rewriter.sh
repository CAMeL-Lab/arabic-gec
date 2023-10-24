#!/usr/bin/env bash
#SBATCH -p nvidia
# use gpus
#SBATCH --gres=gpu:v100:1
# memory
# SBATCH --mem=200GB
# Walltime format hh:mm:ss
#SBATCH --time=40:00:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err


# train_file=/home/ba63/gec-release/data/ged/areta_tags/qalb14/qalb14_train.areta+.nopnx.txt
# test_file=/home/ba63/gec-release/data/ged/areta_tags/qalb14/qalb14_dev.areta+.txt
# ged_model=/scratch/ba63/gec/models/ged++/qalb14/full/wo_camelira/checkpoint-4000
# output_path=/home/ba63/gec-release/gec/outputs/qalb14/mle/qalb14_dev.preds.txt

# train_file=/home/ba63/gec-release/data/ged/areta_tags/qalb14-15/qalb14-15_train.areta+.nopnx.txt
# test_file=/home/ba63/gec-release/data/ged/areta_tags/qalb15/qalb15_dev.areta+.txt
# ged_model=/scratch/ba63/gec/models/ged++/qalb14-15/full/wo_camelira/checkpoint-3000
# output_path=/home/ba63/gec-release/gec/outputs/qalb15/mle/qalb15_dev.preds.txt

# train_file=/home/ba63/gec-release/data/ged/areta_tags/mix/mix_train.areta+.nopnx.txt
# test_file=/home/ba63/gec-release/data/ged/areta_tags/zaebuc/zaebuc_dev.areta+.txt
# ged_model=/scratch/ba63/gec/models/ged++/mix/full/wo_camelira/checkpoint-3000
# output_path=/home/ba63/gec-release/gec/outputs/zaebuc/mle/zaebuc_dev.preds.txt


# train_file=/home/ba63/gec-release/data/ged/areta_tags_camelira/qalb14/qalb14_train.areta+.nopnx.txt
# test_file=/home/ba63/gec-release/data/ged/areta_tags_camelira/qalb14/qalb14_dev.areta+.txt
# ged_model=/scratch/ba63/gec/models/ged++/qalb14/full/w_camelira/checkpoint-4000
# output_path=/home/ba63/gec-release/gec/outputs/qalb14/mle+morph/qalb14_dev.preds.txt

# train_file=/home/ba63/gec-release/data/ged/areta_tags_camelira/qalb14-15/qalb14-15_train.areta+.nopnx.txt
# test_file=/home/ba63/gec-release/data/ged/areta_tags_camelira/qalb15/qalb15_dev.areta+.txt
# ged_model=/scratch/ba63/gec/models/ged++/qalb14-15/full/w_camelira/checkpoint-4500
# output_path=/home/ba63/gec-release/gec/outputs/qalb15/mle+morph/qalb15_dev.preds.txt


train_file=/home/ba63/gec-release/data/ged/areta_tags_camelira/mix/mix_train.areta+.nopnx.txt
test_file=/home/ba63/gec-release/data/ged/areta_tags_camelira/zaebuc/zaebuc_dev.areta+.txt
ged_model=/scratch/ba63/gec/models/ged++/mix/full/w_camelira/checkpoint-5500
output_path=/home/ba63/gec-release/gec/outputs/zaebuc/mle+morph/zaebuc_dev.preds.txt

python rewriter.py \
        --train_file $train_file \
        --test_file  $test_file \
        --ged_model  $ged_model \
        --mode full \
        --cbr_ngrams 2 \
        --output_path $output_path \
        --do_error_ana
