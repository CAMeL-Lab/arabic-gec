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

model_dir=/scratch/ba63/gec/models/gec/qalb14/t5_w_ged
raw_src=/scratch/ba63/gec/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/tune/QALB-2014-L1-Tune.sent.no_ids.clean.dediac

for checkpoint in $model_dir $model_dir/checkpoint-*
do
    if [ ! -d "$checkpoint/ranking" ]; then
        mkdir $checkpoint/ranking
    fi

    for hyp in {1..5}
    do
        tgt=$checkpoint/qalb14_tune.preds.check.${hyp}.txt.pp

        printf "Creating Alignment for $tgt\n"

        python /home/ba63/gec/alignment/aligner.py \
            --src $raw_src \
            --tgt  $tgt \
            --output $checkpoint/ranking/src.to.${hyp}.alignment
    done
done