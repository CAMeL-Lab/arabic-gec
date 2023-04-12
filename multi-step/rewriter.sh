#!/usr/bin/env bash
#SBATCH -p nvidia
#SBATCH -q nlp
# use gpus
#SBATCH --gres=gpu:1
# memory
#SBATCH --mem=300GB
# Walltime format hh:mm:ss
#SBATCH --time=40:00:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err

train_file=/scratch/ba63/gec/data/
output_path=outputs/CBR/w_camelira/qalb14_dev.preds.txt

# --ged_model /scratch/ba63/gec/models/ged++/qalb14/wo_camelira/checkpoint-1500 \
# --train_file /scratch/ba63/gec/data/alignment/modeling_areta_tags_improved/qalb14/qalb14_train.areta+.nopnx.txt \
# --test_file  /scratch/ba63/gec/data/alignment/modeling_areta_tags_improved/qalb14/qalb14_dev.areta+.txt \

python rewriter.py \
        --train_file /scratch/ba63/gec/data/gec_camelira/areta_tags/qalb14/qalb14_train.areta+.nopnx.txt \
        --test_file /scratch/ba63/gec/data/gec_camelira/areta_tags/qalb14/qalb14_dev.areta+.txt \
        --ged_model /scratch/ba63/gec/models/ged++/qalb14/w_camelira/checkpoint-1500 \
        --cbr_ngrams 2 \
        --output_path $output_path \
        --do_error_ana
