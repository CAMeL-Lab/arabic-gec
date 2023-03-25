#!/bin/bash
# Set number of tasks to run
#SBATCH -p nvidia
#SBATCH -q nlp
# SBATCH --ntasks=1
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
m2_edits=/scratch/ba63/gec/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/tune/QALB-2014-L1-Tune.m2

sys=/home/ba63/gec/multi-step/outputs/CBR+T5
eval_file=cbr+t5_w_camelira.txt

# sys1=/scratch/ba63/gec/models/gec/qalb14/t5_w_camelira_ged
# eval_file=qalb14_tune.no_oracle.preds.txt.pp

for checkpoint in ${sys} ${sys}/checkpoint-* # ${sys2} ${sys2}/checkpoint-*
    do
        printf "Evaluating ${checkpoint}\n"
        python $m2_scorer $checkpoint/$eval_file $m2_edits > $checkpoint/m2.$eval_file.eval
        # python $m2_scorer --beta 0.5 $checkpoint/$eval_file $m2_edits > $checkpoint/m2.$eval_file.f0.5 #F0.5 eval
        # cat $checkpoint/m2.$eval_file.f0.5 | grep "F" >> $checkpoint/m2.$eval_file.eval
        # rm $checkpoint/m2.$eval_file.f0.5
    done
