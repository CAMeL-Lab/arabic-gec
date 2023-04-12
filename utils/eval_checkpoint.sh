#!/bin/bash
# Set number of tasks to run
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
m2_edits=/scratch/ba63/gec/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/dev/QALB-2014-L1-Dev.m2
# m2_edits=/scratch/ba63/gec/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/dev/QALB-2015-L2-Dev.m2
# m2_edits=/scratch/ba63/gec/data/alignment/m2_files/zaebuc_dev.m2
# m2_edits=/scratch/ba63/gec/data/alignment/m2_files/mix_tune.m2


sys=/scratch/ba63/gec/models/gec/qalb14/bart
eval_file=qalb14_dev.preds.txt.pp


for checkpoint in ${sys} # ${sys}/checkpoint-*
    do
        printf "Evaluating ${checkpoint}\n"
        python $m2_scorer $checkpoint/$eval_file $m2_edits > $checkpoint/m2.$eval_file.eval
    done