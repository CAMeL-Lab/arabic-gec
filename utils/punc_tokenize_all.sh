#!/bin/bash
# Set number of tasks to run
#SBATCH -q nlp
# SBATCH --ntasks=1
# Set number of cores per task (default is 1)
#SBATCH --cpus-per-task=1
# Walltime format hh:mm:ss
#SBATCH --time=47:59:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err

model=t5
exp=mix
gran=full

# out1=/scratch/ba63/gec/models/gec/$exp/$gran/${model}
out2=/scratch/ba63/gec/models/gec/$exp/$gran/${model}_w_ged
# out3=/scratch/ba63/gec/models/gec/$exp/$gran/${model}_w_ged_pred_worst
# out4=/scratch/ba63/gec/models/gec/$exp/$gran/${model}_w_camelira
out5=/scratch/ba63/gec/models/gec/$exp/$gran/${model}_w_camelira_ged
# out6=/scratch/ba63/gec/models/gec/$exp/$gran/${model}_w_camelira_ged_pred_worst

for OUTPUT_DIR in $out2 $out5 # $out3 $out4 $out5 $out6
do
    for f in ${OUTPUT_DIR} ${OUTPUT_DIR}/checkpoint*
        do
            printf "Punctuation tokenizing $f\n"
            python /home/ba63/gec/bart-t5-new/utils/punc_tokenize.py \
                --input $f/zaebuc_dev.preds.oracle.txt \
                --output $f/zaebuc_dev.preds.oracle.txt.punc_tokenized
            printf "\n\n"
            mv $f/zaebuc_dev.preds.oracle.txt.punc_tokenized $f/zaebuc_dev.preds.oracle.txt
        done
done