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


# output_path=outputs/CBR/cbr.oracle.txt
# --ged_model /scratch/ba63/gec/models/ged/qalb14/wo_camelira/checkpoint-1500 \

# python rewriter.py \
#     --train_file /scratch/ba63/gec/data/alignment/modeling_areta_tags_improved/qalb14/qalb14_train.areta+.nopnx.txt \
#     --test_file  /scratch/ba63/gec/data/alignment/modeling_areta_tags_improved/qalb14/qalb14_tune.areta+.txt \
#     --cbr_ngrams 2 \
#     --output_path $output_path




# output_path=outputs/CBR+T5/cbr+t5.oracle.txt
# --ged_model /scratch/ba63/gec/models/ged/qalb14/wo_camelira/checkpoint-1500 \
# python rewriter.py \
#     --train_file /scratch/ba63/gec/data/alignment/modeling_areta_tags_improved/qalb14/qalb14_train.areta+.nopnx.txt \
#     --test_file  /scratch/ba63/gec/data/alignment/modeling_areta_tags_improved/qalb14/qalb14_tune.areta+.txt \
#     --cbr_ngrams 2 \
#     --seq2seq_model /scratch/ba63/gec/models/gec/qalb14/t5 \
#     --output_path $output_path \
#     --do_error_ana


# output_path=outputs/CBR/cbr_w_camelira.txt
# --ged_model /scratch/ba63/gec/models/ged/qalb14/w_camelira/checkpoint-1500 \

# python rewriter.py \
#     --train_file /scratch/ba63/gec/data/gec_camelira/areta_tags/qalb14_train.areta+.txt.nopnx \
#     --test_file  /scratch/ba63/gec/data/gec_camelira/areta_tags/qalb14_tune.areta+.txt \
#     --cbr_ngrams 2 \
#     --ged_model /scratch/ba63/gec/models/ged/qalb14/w_camelira/checkpoint-1500 \
#     --output_path $output_path


output_path=outputs/CBR+T5/cbr+t5_w_camelira.txt
# --ged_model /scratch/ba63/gec/models/ged/qalb14/w_camelira/checkpoint-1500 \

python rewriter.py \
    --train_file /scratch/ba63/gec/data/gec_camelira/areta_tags/qalb14_train.areta+.txt.nopnx \
    --test_file  /scratch/ba63/gec/data/gec_camelira/areta_tags/qalb14_tune.areta+.txt \
    --cbr_ngrams 2 \
    --ged_model /scratch/ba63/gec/models/ged/qalb14/w_camelira/checkpoint-1500 \
    --seq2seq_model /scratch/ba63/gec/models/gec/qalb14/t5_w_camelira \
    --output_path $output_path \
    --do_error_ana
