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


python get_morph_features.py \
    --input /scratch/ba63/gec/data/alignment/modeling_areta_tags_improved/qalb14/qalb14_train.areta+.txt \
    --output /scratch/ba63/gec/data/synthetic/syn-data-utils/qalb14_train.areta+.morph.txt
 

python get_morph_features.py \
    --input /scratch/ba63/gec/data/alignment/modeling_areta_tags_improved/qalb14/qalb14_tune.areta+.txt \
    --output /scratch/ba63/gec/data/synthetic/syn-data-utils/qalb14_tune.areta+.morph.txt

