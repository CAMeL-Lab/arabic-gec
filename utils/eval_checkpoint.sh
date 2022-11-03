#!/bin/bash
#SBATCH -q nlp
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

m2_scorer=/scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/m2Scripts/m2scorer.py
# m2_edits=/scratch/ba63/gec/data/MIX/dev.m2
# m2_edits=/scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/dev/QALB-2015-L2-Dev.m2
m2_edits=/scratch/ba63/gec/data/ZAEBUC-v1.0/data/ar/dev/m2.dev.edits
system_hyp=/scratch/ba63/gec/models/QALB-2014/t5_lr_with_pref_30
eval_file=dev.zaebuc.pred.txt.pnx.edits

if [ "$1" = "all" ]; then

 for checkpoint in ${system_hyp} ${system_hyp}/checkpoint*
     do
         printf "Evaluating ${checkpoint}\n"
         python $m2_scorer $checkpoint/$eval_file $m2_edits > $checkpoint/m2.$eval_file.eval #F1 eval
         python $m2_scorer --beta 0.5 $checkpoint/$eval_file $m2_edits > $checkpoint/m2.$eval_file.f5.0 #F0.5 eval
         cat $checkpoint/m2.$eval_file.f5.0 | grep "F" >> $checkpoint/m2.$eval_file.eval
         rm $checkpoint/m2.$eval_file.f5.0
     done
else
         printf "Evaluating ${system_hyp}\n"
         python $m2_scorer $system_hyp/$eval_file $m2_edits > $system_hyp/m2.$eval_file.eval #F1 eval
         python $m2_scorer --beta 0.5 $system_hyp/$eval_file $m2_edits > $system_hyp/m2.$eval_file.f5.0 #F0.5 eval
         cat $system_hyp/m2.$eval_file.f5.0 | grep "F" >> $system_hyp/m2.$eval_file.eval
         rm $system_hyp/m2.$eval_file.f5.0
fi
