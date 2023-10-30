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

m2_scorer=/home/ba63/gec-release/gec/utils/evaluate.py

# m2_edits=/scratch/ba63/gec/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/dev/QALB-2014-L1-Dev.m2
# m2_edits=/scratch/ba63/gec/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/dev/QALB-2015-L2-Dev.m2
m2_edits=/home/ba63/gec-release/data/m2edits/zaebuc/zaebuc_dev.m2


sys=/home/ba63/gec-release/gec/outputs/zaebuc/mle+morph
eval_file=zaebuc_dev.preds.txt

printf "Evaluating ${sys}\n"

python $m2_scorer \
    --system_output $sys/$eval_file \
    --m2_file $m2_edits