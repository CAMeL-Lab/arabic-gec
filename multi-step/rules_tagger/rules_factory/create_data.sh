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

# TODO: Note you have to fix the randomness when applying UNK tokens to training!!

train_alignment=/scratch/ba63/gec/data/alignment/modeling_areta_tags_improved/qalb14/qalb14_train.areta+.txt
tune_alignment=/scratch/ba63/gec/data/alignment/modeling_areta_tags_improved/qalb14/qalb14_tune.areta+.txt
output_dir=/scratch/ba63/gec/data/rules-tagger-data/wo_camelira

echo "Generating data for $train_alignment using all rules"

python rules.py \
    --alignment_file $train_alignment \
    --rules_file $output_dir/all/rules.txt \
    --mode train \
    --output_file $output_dir/all/train.txt

python rules.py \
    --alignment_file $tune_alignment \
    --rules_file $output_dir/all/rules.txt \
    --mode tune \
    --output_file $output_dir/all/tune.txt

for n in {4,5,10}
do
    echo "Generating data for $train_alignment using $n rules"

    python rules.py \
        --alignment_file $train_alignment \
        --rules_file $output_dir/$n/rules.txt \
        --mode train \
        --prune_rules $n \
        --output_file $output_dir/$n/train.txt

    echo "Generating data for $train_alignment using $n rules"

    python rules.py \
        --alignment_file $tune_alignment \
        --rules_file $output_dir/$n/rules.txt \
        --mode tune \
        --output_file $output_dir/$n/tune.txt
done


train_alignment=/scratch/ba63/gec/data/gec_camelira/areta_tags/qalb14_train.areta+.txt
tune_alignment=/scratch/ba63/gec/data/gec_camelira/areta_tags/qalb14_tune.areta+.txt
output_dir=/scratch/ba63/gec/data/rules-tagger-data/w_camelira

for n in {4,5,10}
do
    echo "Generating data for $train_alignment using $n rules"

    python rules.py \
        --alignment_file $train_alignment \
        --rules_file $output_dir/$n/rules.txt \
        --mode train \
        --prune_rules $n \
        --output_file $output_dir/$n/train.txt


    echo "Generating data for $tune_alignment using $n rules"


    python rules.py \
        --alignment_file $tune_alignment \
        --rules_file $output_dir/$n/rules.txt \
        --mode tune \
        --output_file $output_dir/$n/tune.txt
done


echo "Generating data for $train_alignment using all rules"

python rules.py \
    --alignment_file $train_alignment \
    --rules_file $output_dir/all/rules.txt \
    --mode train \
    --output_file $output_dir/all/train.txt

python rules.py \
    --alignment_file $tune_alignment \
    --rules_file $output_dir/all/rules.txt \
    --mode tune \
    --output_file $output_dir/all/tune.txt
