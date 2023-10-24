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


qalb14_dir=/home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/dev
qalb15_dir=/home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/dev

python  m2scorer/scripts/edit_creator.py \
        $qalb14_dir/QALB-2014-L1-Dev.sent.no_ids \
        $qalb14_dir/QALB-2014-L1-Dev.cor.no_ids \
        > qalb14_dev.m2

python  m2scorer/scripts/edit_creator.py \
        $qalb15_dir/QALB-2015-L2-Dev.sent.no_ids \
        $qalb15_dir/QALB-2015-L2-Dev.cor.no_ids \
        > qalb15_dev.m2
