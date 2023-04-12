#!/bin/bash
# Set number of tasks to run
#SBATCH --ntasks=1
# Set number of cores per task (default is 1)
#SBATCH --cpus-per-task=10
# memory
#SBATCH --mem=120GB
# Walltime format hh:mm:ss
#SBATCH --time=47:59:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err

eval "$(conda shell.bash hook)"
conda activate python2

m2_scorer=/scratch/ba63/gec/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/m2Scripts/m2scorer.py
# m2_edits=/scratch/ba63/gec/data/alignment/m2_files/qalb14_tune.nopnx.m2
# m2_edits=/scratch/ba63/gec/data/alignment/m2_files/qalb15_dev.nopnx.m2
m2_edits=/scratch/ba63/gec/data/alignment/m2_files/zaebuc_dev.nopnx.m2
# m2_edits=/scratch/ba63/gec/data/alignment/m2_files/mix_tune.nopnx.m2


sys=/scratch/ba63/gec/models/gec/mix/t5_w_camelira_ged_pred_worst
eval_file=zaebuc_dev.preds.merge_fix.txt.nopnx


for checkpoint in ${sys} ${sys}/checkpoint-*

    do
        printf "Evaluating ${checkpoint}\n"
        python $m2_scorer $checkpoint/$eval_file $m2_edits > $checkpoint/m2.$eval_file.eval #F1 eval
        # python $m2_scorer --beta 0.5 $checkpoint/$eval_file.nopnx $m2_edits > $checkpoint/m2.$eval_file.nopnx.f0.5 #F0.5 eval
        # cat $checkpoint/m2.$eval_file.nopnx.f0.5 | grep "F" >> $checkpoint/m2.$eval_file.nopnx.eval
        # rm $checkpoint/m2.$eval_file.nopnx.f0.5
    done


