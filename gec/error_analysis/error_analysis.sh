#!/bin/bash

# Generating alignment
# src=/home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/dev/QALB-2014-L1-Dev.sent.no_ids.clean.dediac
# system_output=/scratch/ba63/gec/models/gec/qalb14/full/bart/checkpoint-3000/qalb14_dev.preds.check.txt.pp
# error_analysis_dir=/home/ba63/gec-release/error_analysis/outputs/bart_correct/qalb14

# src=/home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/dev/QALB-2015-L2-Dev.sent.no_ids.dediac
# system_output=/scratch/ba63/gec/models/gec/qalb14-15/full/bart/checkpoint-11000/qalb15_dev.preds.check.txt.pp
# error_analysis_dir=/home/ba63/gec-release/error_analysis/outputs/bart_correct/qalb15

# src=/home/ba63/gec-release/data/gec/ZAEBUC-v1.0/data/ar/dev/dev.sent.raw.pnx.tok.dediac
# system_output=/scratch/ba63/gec/models/gec/mix/full/bart/checkpoint-8000/zaebuc_dev.preds.check.txt
# error_analysis_dir=/home/ba63/gec-release/error_analysis/outputs/bart_correct/zaebuc


# src=/home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/dev/QALB-2014-L1-Dev.sent.no_ids.clean.dediac
# system_output=/scratch/ba63/gec/models/gec/qalb14/coarse/bart_w_camelira_ged_pred_worst/checkpoint-3000/qalb14_dev.preds.check.txt.pp
# error_analysis_dir=/home/ba63/gec-release/error_analysis/outputs/bart_w_camelira_ged_pred_worst_correct/qalb14

# src=/home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/dev/QALB-2015-L2-Dev.sent.no_ids.dediac
# system_output=/scratch/ba63/gec/models/gec/qalb14-15/coarse/bart_w_camelira_ged_pred_worst/qalb15_dev.preds.check.txt.pp
# error_analysis_dir=/home/ba63/gec-release/error_analysis/outputs/bart_w_camelira_ged_pred_worst_correct/qalb15

src=/home/ba63/gec-release/data/gec/ZAEBUC-v1.0/data/ar/dev/dev.sent.raw.pnx.tok.dediac
system_output=/scratch/ba63/gec/models/gec/mix/coarse/bart_w_camelira_ged_pred_worst/checkpoint-11000/zaebuc_dev.preds.check.txt
error_analysis_dir=/home/ba63/gec-release/error_analysis/outputs/bart_w_camelira_ged_pred_worst_correct/zaebuc



alignment_output=${error_analysis_dir}/qalb15_dev.alignment.txt

printf "Generating alignments for ${src}..\n"

python /home/ba63/gec-release/alignment/aligner.py \
    --src ${src} \
    --tgt ${system_output} \
    --output ${alignment_output}


# Getting the areta tags
cd /home/ba63/gec-release/areta

eval "$(conda shell.bash hook)"
conda activate areta

areta_tags_output=${error_analysis_dir}/zaebuc_dev.areta.txt
enriched_areta_tags_output=${error_analysis_dir}/zaebuc_dev.areta+.txt

printf "Generating areta tags for ${alignment_output}..\n"

python /home/ba63/gec-release/areta/annotate_err_type_ar.py \
    --alignment $alignment_output \
    --output_path $areta_tags_output \
    --enriched_output_path $enriched_areta_tags_output

rm fout2.basic



# Converting the areta tags to whatever ged scheme we want
python /home/ba63/gec-release/data/ged/create_ged_data.py \
    --input $enriched_areta_tags_output \
    --output ${error_analysis_dir}/zaebuc_dev.ged.txt


python /home/ba63/gec-release/data/ged/granularity_map.py \
    --input ${error_analysis_dir}/zaebuc_dev.ged.txt \
    --mode coarse \
    --output ${error_analysis_dir}/zaebuc_dev.ged.coarse.txt
 

# Evaluation
# labels=/home/ba63/gec-release/data/ged/qalb14/wo_camelira/coarse/labels.txt
labels=/home/ba63/gec-release/data/ged/mix/w_camelira/coarse/labels.txt
gold_data=/home/ba63/gec-release/data/ged/zaebuc/wo_camelira/coarse/dev.txt

system=${error_analysis_dir}/zaebuc_dev.ged.coarse.txt

paste $gold_data $system | cut -f1,2,4 > eval_data


python /home/ba63/gec-release/ged/evaluate.py \
    --data eval_data \
    --labels $labels \
    --output $system.results

rm eval_data