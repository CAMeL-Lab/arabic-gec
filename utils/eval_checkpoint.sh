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


m2_scorer=/scratch/ba63/gec/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/m2Scripts/m2scorer.py
# m2_edits=/scratch/ba63/gec/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/test/QALB-2014-L1-Test.m2
m2_edits=/scratch/ba63/gec/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/test/QALB-2015-L2-Test.m2
# m2_edits=/scratch/ba63/gec/data/alignment/m2_files/zaebuc_test.m2
# m2_edits=/scratch/ba63/gec/data/alignment/m2_files/mix_tune.m2

# sys1=/scratch/ba63/gec/models/gec/qalb14/full/bart_w_ged/checkpoint-3000
# sys2=/scratch/ba63/gec/models/gec/qalb14/full/bart_w_ged_pred_worst/checkpoint-3000
# sys1=/scratch/ba63/gec/models/gec/qalb14/full/bart_w_camelira_ged/checkpoint-3500
# sys2=/scratch/ba63/gec/models/gec/qalb14/full/bart_w_camelira_ged_pred_worst/checkpoint-3000
# sys1=/scratch/ba63/gec/models/gec/qalb14/full/t5_w_ged/checkpoint-36000
# sys2=/scratch/ba63/gec/models/gec/qalb14/full/t5_w_ged_pred_worst/checkpoint-34500
# sys1=/scratch/ba63/gec/models/gec/qalb14/full/t5_w_camelira_ged/checkpoint-36000
# sys2=/scratch/ba63/gec/models/gec/qalb14/full/t5_w_camelira_ged_pred_worst/checkpoint-33000

# sys1=/scratch/ba63/gec/models/gec/qalb14/coarse/bart_w_ged/checkpoint-3000
# sys2=/scratch/ba63/gec/models/gec/qalb14/coarse/bart_w_ged_pred_worst/checkpoint-3000
# sys1=/scratch/ba63/gec/models/gec/qalb14/coarse/bart_w_camelira_ged/checkpoint-5500
# sys2=/scratch/ba63/gec/models/gec/qalb14/coarse/bart_w_camelira_ged_pred_worst/checkpoint-3000
# sys1=/scratch/ba63/gec/models/gec/qalb14/coarse/t5_w_ged/checkpoint-25500
# sys2=/scratch/ba63/gec/models/gec/qalb14/coarse/t5_w_ged_pred_worst/checkpoint-34500
# sys1=/scratch/ba63/gec/models/gec/qalb14/coarse/t5_w_camelira_ged/checkpoint-21000
# sys2=/scratch/ba63/gec/models/gec/qalb14/coarse/t5_w_camelira_ged_pred_worst/checkpoint-36000 -------------------------> [BUG]

# sys1=/scratch/ba63/gec/models/gec/qalb14/binary/bart_w_ged/checkpoint-3000
# sys2=/scratch/ba63/gec/models/gec/qalb14/binary/bart_w_ged_pred_worst/checkpoint-3000
# sys1=/scratch/ba63/gec/models/gec/qalb14/binary/bart_w_camelira_ged/checkpoint-3500
# sys2=/scratch/ba63/gec/models/gec/qalb14/binary/bart_w_camelira_ged_pred_worst/checkpoint-6000
# sys1=/scratch/ba63/gec/models/gec/qalb14/binary/t5_w_ged/checkpoint-36000
# sys2=/scratch/ba63/gec/models/gec/qalb14/binary/t5_w_ged_pred_worst/checkpoint-21000
# sys1=/scratch/ba63/gec/models/gec/qalb14/binary/t5_w_camelira_ged/checkpoint-31500
# sys2=/scratch/ba63/gec/models/gec/qalb14/binary/t5_w_camelira_ged_pred_worst/checkpoint-22500 -------------------------> [BUG]


# sys1=/scratch/ba63/gec/models/gec/qalb14-15/full/bart_w_ged/checkpoint-6000
# sys2=/scratch/ba63/gec/models/gec/qalb14-15/full/bart_w_ged_pred_worst/checkpoint-10000
# sys1=/scratch/ba63/gec/models/gec/qalb14-15/full/bart_w_camelira_ged
# sys2=/scratch/ba63/gec/models/gec/qalb14-15/full/bart_w_camelira_ged_pred_worst/checkpoint-11000
# sys1=/scratch/ba63/gec/models/gec/qalb14-15/full/t5_w_ged/checkpoint-66000
# sys2=/scratch/ba63/gec/models/gec/qalb14-15/full/t5_w_ged_pred_worst/checkpoint-57000
# sys1=/scratch/ba63/gec/models/gec/qalb14-15/full/t5_w_camelira_ged/checkpoint-69000
# sys2=/scratch/ba63/gec/models/gec/qalb14-15/full/t5_w_camelira_ged_pred_worst/checkpoint-72000


# sys1=/scratch/ba63/gec/models/gec/qalb14-15/coarse/bart_w_ged/checkpoint-3000
# sys2=/scratch/ba63/gec/models/gec/qalb14-15/coarse/bart_w_ged_pred_worst/checkpoint-10000
# sys1=/scratch/ba63/gec/models/gec/qalb14-15/coarse/bart_w_camelira_ged/checkpoint-6000
# sys2=/scratch/ba63/gec/models/gec/qalb14-15/coarse/bart_w_camelira_ged_pred_worst
# sys1=/scratch/ba63/gec/models/gec/qalb14-15/coarse/t5_w_ged/checkpoint-54000 ---------------------------------> [BUG]
# sys2=/scratch/ba63/gec/models/gec/qalb14-15/coarse/t5_w_ged_pred_worst/checkpoint-63000
# sys1=/scratch/ba63/gec/models/gec/qalb14-15/coarse/t5_w_camelira_ged/checkpoint-66000
# sys2=/scratch/ba63/gec/models/gec/qalb14-15/coarse/t5_w_camelira_ged_pred_worst/checkpoint-66000


# sys1=/scratch/ba63/gec/models/gec/qalb14-15/binary/bart_w_ged
# sys2=/scratch/ba63/gec/models/gec/qalb14-15/binary/bart_w_ged_pred_worst/checkpoint-10000
# sys1=/scratch/ba63/gec/models/gec/qalb14-15/binary/bart_w_camelira_ged/checkpoint-10000
# sys2=/scratch/ba63/gec/models/gec/qalb14-15/binary/bart_w_camelira_ged_pred_worst
# sys1=/scratch/ba63/gec/models/gec/qalb14-15/binary/t5_w_ged/checkpoint-54000 ---------------------------------> [BUG]
# sys2=/scratch/ba63/gec/models/gec/qalb14-15/binary/t5_w_ged_pred_worst/checkpoint-48000
# sys1=/scratch/ba63/gec/models/gec/qalb14-15/binary/t5_w_camelira_ged/checkpoint-60000
# sys2=/scratch/ba63/gec/models/gec/qalb14-15/binary/t5_w_camelira_ged_pred_worst


# sys1=/scratch/ba63/gec/models/gec/mix/full/bart_w_ged/checkpoint-11000
# sys2=/scratch/ba63/gec/models/gec/mix/full/bart_w_ged_pred_worst/checkpoint-9000
# sys1=/scratch/ba63/gec/models/gec/mix/full/bart_w_camelira_ged/checkpoint-9000
# sys2=/scratch/ba63/gec/models/gec/mix/full/bart_w_camelira_ged_pred_worst/checkpoint-11000
# sys1=/scratch/ba63/gec/models/gec/mix/full/t5_w_ged/checkpoint-48000
# sys2=/scratch/ba63/gec/models/gec/mix/full/t5_w_ged_pred_worst/checkpoint-24000
# sys1=/scratch/ba63/gec/models/gec/mix/full/t5_w_camelira_ged/checkpoint-39000
# sys2=/scratch/ba63/gec/models/gec/mix/full/t5_w_camelira_ged_pred_worst/checkpoint-39000

# sys1=/scratch/ba63/gec/models/gec/mix/coarse/bart_w_ged/checkpoint-4000
# sys2=/scratch/ba63/gec/models/gec/mix/coarse/bart_w_ged_pred_worst/checkpoint-9000
# sys1=/scratch/ba63/gec/models/gec/mix/coarse/bart_w_camelira_ged
# sys2=/scratch/ba63/gec/models/gec/mix/coarse/bart_w_camelira_ged_pred_worst/checkpoint-11000
# sys1=/scratch/ba63/gec/models/gec/mix/coarse/t5_w_ged/checkpoint-24000
# sys2=/scratch/ba63/gec/models/gec/mix/coarse/t5_w_ged_pred_worst/checkpoint-60000
# sys1=/scratch/ba63/gec/models/gec/mix/coarse/t5_w_camelira_ged/checkpoint-27000
# sys2=/scratch/ba63/gec/models/gec/mix/coarse/t5_w_camelira_ged_pred_worst/checkpoint-51000

# sys1=/scratch/ba63/gec/models/gec/mix/binary/bart_w_ged/checkpoint-11000
# sys2=/scratch/ba63/gec/models/gec/mix/binary/bart_w_ged_pred_worst/checkpoint-11000
# sys1=/scratch/ba63/gec/models/gec/mix/binary/bart_w_camelira_ged/checkpoint-10000
# sys2=/scratch/ba63/gec/models/gec/mix/binary/bart_w_camelira_ged_pred_worst/checkpoint-11000
# sys1=/scratch/ba63/gec/models/gec/mix/binary/t5_w_ged/checkpoint-33000
# sys2=/scratch/ba63/gec/models/gec/mix/binary/t5_w_ged_pred_worst/checkpoint-63000
# sys1=/scratch/ba63/gec/models/gec/mix/binary/t5_w_camelira_ged/checkpoint-24000
# sys2=/scratch/ba63/gec/models/gec/mix/binary/t5_w_camelira_ged_pred_worst/checkpoint-48000

# output=zaebuc_dev.preds.txt


# for sys in $sys1 $sys2
# do
#     printf "Evaluation $sys \n"
#     python $m2_scorer $sys/$output $m2_edits > $sys/$output.official.m2
# done

sys=/scratch/ba63/gec/models/gec/qalb14-15/coarse/bart_w_camelira_ged_pred_worst
output=qalb15_test_L2.preds.txt

printf "Evaluation $sys \n"
python $m2_scorer $sys/$output $m2_edits > $sys/$output.official.m2
