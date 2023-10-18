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


###QALB-2014 GED MODELS####
# Qalb-2014:
# /scratch/ba63/gec/models/ged++/qalb14/full/wo_camelira/checkpoint-4000
# /scratch/ba63/gec/models/ged++/qalb14/full/w_camelira/checkpoint-4000
# /scratch/ba63/gec/models/ged++/qalb14/coarse/wo_camelira
# /scratch/ba63/gec/models/ged++/qalb14/coarse/w_camelira/checkpoint-1500
# /scratch/ba63/gec/models/ged++/qalb14/binary/wo_camelira/checkpoint-500
# /scratch/ba63/gec/models/ged++/qalb14/binary/w_camelira/checkpoint-1000


# Qalb-2014-15:
# /scratch/ba63/gec/models/ged++/qalb14-15/wo_camelira/checkpoint-3000
# /scratch/ba63/gec/models/ged++/qalb14-15/w_camelira/checkpoint-1000


# MIX:
# /scratch/ba63/gec/models/ged++/mix/wo_camelira/checkpoint-2000
# /scratch/ba63/gec/models/ged++/mix/w_camelira/checkpoint-2500


###QALB-2015 GED MODELS####
# Qalb-2014:
# /scratch/ba63/gec/models/ged++/qalb14/full/wo_camelira/checkpoint-5000
# /scratch/ba63/gec/models/ged++/qalb14/full/w_camelira/checkpoint-2000


# Qalb-2014-15:
# /scratch/ba63/gec/models/ged++/qalb14-15/full/wo_camelira/checkpoint-3000
# /scratch/ba63/gec/models/ged++/qalb14-15/full/w_camelira/checkpoint-4500
# /scratch/ba63/gec/models/ged++/qalb14-15/coarse/wo_camelira/checkpoint-3000
# /scratch/ba63/gec/models/ged++/qalb14-15/coarse/w_camelira/checkpoint-6000
# /scratch/ba63/gec/models/ged++/qalb14-15/binary/wo_camelira/checkpoint-2000
# /scratch/ba63/gec/models/ged++/qalb14-15/binary/w_camelira/checkpoint-6000


# MIX:
# /scratch/ba63/gec/models/ged++/mix/wo_camelira/checkpoint-2000
# /scratch/ba63/gec/models/ged++/mix/w_camelira/checkpoint-5000


###ZAEBUC GED MODELS####
# Qalb-2014:
# /scratch/ba63/gec/models/ged++/qalb14/full/wo_camelira/checkpoint-3000
# /scratch/ba63/gec/models/ged++/qalb14/full/w_camelira/checkpoint-2000


# Qalb-2014-15:
# /scratch/ba63/gec/models/ged++/qalb14-15/wo_camelira/checkpoint-2500
# /scratch/ba63/gec/models/ged++/qalb14-15/w_camelira/checkpoint-1500


# MIX:
# /scratch/ba63/gec/models/ged++/mix/full/wo_camelira/checkpoint-3000
# /scratch/ba63/gec/models/ged++/mix/full/w_camelira/checkpoint-5500
# /scratch/ba63/gec/models/ged++/mix/coarse/wo_camelira/checkpoint-3500
# /scratch/ba63/gec/models/ged++/mix/coarse/w_camelira/checkpoint-4500
# /scratch/ba63/gec/models/ged++/mix/binary/wo_camelira/checkpoint-4500
# /scratch/ba63/gec/models/ged++/mix/binary/w_camelira/checkpoint-5000


# train_file=/scratch/ba63/gec/data/alignment/modeling_areta_tags_improved/binary/mix/train.areta+.nopnx.txt
# test_file=/scratch/ba63/gec/data/alignment/modeling_areta_tags_improved/binary/zaebuc/zaebuc_dev.areta+.txt

camelira_train_file=/scratch/ba63/gec/data/gec_camelira/areta_tags/binary/mix/train.areta+.nopnx.txt
camelira_test_file=/scratch/ba63/gec/data/gec_camelira/areta_tags/binary/zaebuc/zaebuc_dev.areta+.txt

# ged_model=/scratch/ba63/gec/models/ged++/mix/binary/wo_camelira/checkpoint-4500
# output_path=outputs/CBR/zaebuc/mix/binary/wo_camelira/zaebuc_dev.preds.txt

ged_model=/scratch/ba63/gec/models/ged++/mix/binary/w_camelira/checkpoint-5000
output_path=outputs/CBR/zaebuc/mix/binary/w_camelira/zaebuc_dev.preds.txt

nvidia-smi

python rewriter.py \
        --train_file $camelira_train_file \
        --test_file  $camelira_test_file \
        --ged_model  $ged_model \
        --mode coarse \
        --cbr_ngrams 2 \
        --output_path $output_path \
        --do_error_ana
