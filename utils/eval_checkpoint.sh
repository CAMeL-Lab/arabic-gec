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

eval "$(conda shell.bash hook)"
conda activate python2


m2_scorer=/home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/m2Scripts/m2scorer.py
m2_edits=/scratch/ba63/gec/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/dev/QALB-2015-L2-Dev.m2
# m2_edits_nopnx=/home/ba63/gec-release/data/m2edits/qalb15/qalb15_L1-test.nopnx.m2


# m2_edits=/home/ba63/gec-release/data/m2edits/zaebuc/zaebuc_dev.m2
# m2_edits_nopnx=/home/ba63/gec-release/data/m2edits/zaebuc/zaebuc_test.nopnx.m2


# sys1=/scratch/ba63/gec/models/gec/qalb14-15/full/bart/checkpoint-7000
# sys2=/scratch/ba63/gec/models/gec/qalb14-15/full/bart_w_camelira/checkpoint-8000
# sys3=/scratch/ba63/gec/models/gec/qalb14-15/full/bart_w_ged_pred_worst/checkpoint-6000

# sys1=/scratch/ba63/gec/models/gec/qalb14-15/full/bart_w_camelira_ged_pred_worst/checkpoint-7000
# sys2=/scratch/ba63/gec/models/gec/qalb14-15/coarse/bart_w_ged_pred_worst/checkpoint-6000
# sys3=/scratch/ba63/gec/models/gec/qalb14-15/coarse/bart_w_camelira_ged_pred_worst/checkpoint-8000

# sys1=/scratch/ba63/gec/models/gec/qalb14-15/binary/bart_w_ged_pred_worst/checkpoint-8000
# sys2=/scratch/ba63/gec/models/gec/qalb14-15/binary/bart_w_camelira_ged_pred_worst/checkpoint-6000

sys1=/home/ba63/gec-release/gec/outputs/qalb15/chatgpt

output=chatgpt_output_3_shot_qalb15_l2_dev.txt.clean

for sys in $sys1 # $sys2 # $sys3
do
    printf "Evaluation $sys \n"
    python $m2_scorer $sys/$output $m2_edits > $sys/$output.official.m2
    # python $m2_scorer $sys/$output.nopnx $m2_edits_nopnx > $sys/$output.nopnx.official.m2

done

