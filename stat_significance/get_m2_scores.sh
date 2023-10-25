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


m2_edits=/scratch/ba63/gec/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/test/QALB-2014-L1-Test.m2
# m2_edits=/scratch/ba63/gec/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/test/QALB-2015-L2-Test.m2

# sys=/scratch/ba63/gec/models/gec/qalb14-15/full/bart_w_camelira_ged_pred_worst/checkpoint-7000/qalb15_test-L1.preds.check.txt

# sys=/scratch/ba63/gec/models/gec/qalb14/full/bart_w_camelira/checkpoint-3500/qalb14_test.preds.check.txt
sys=/scratch/ba63/gec/models/gec/qalb14/full/bart/checkpoint-3000/qalb14_test.preds.check.txt

printf "Evaluating ${sys}\n"

python $m2_scorer \
    --system_output $sys \
    --m2_file $m2_edits \
    --mode single

