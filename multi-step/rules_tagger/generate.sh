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


MODEL_DIR=/scratch/ba63/gec/models/rules_tagger/wo_camelira/all
DATA=/scratch/ba63/gec/data/rules-tagger-data/wo_camelira/all

    for checkpoint in $MODEL_DIR $MODEL_DIR/checkpoint-*
    do

        echo "Generating outputs from $checkpoint.."

        python generate.py \
            --data $DATA/tune.txt.json \
            --pred $checkpoint/qalb14_tune.preds.txt \
            --output $checkpoint/qalb14_tune.preds.gen.txt

    done


for exp in w_camelira wo_camelira
do
    for n in {4,5,10}
    do

        MODEL_DIR=/scratch/ba63/gec/models/rules_tagger/$exp/$n
        DATA=/scratch/ba63/gec/data/rules-tagger-data/$exp/$n

        for checkpoint in $MODEL_DIR $MODEL_DIR/checkpoint-*
        do

            echo "Generating outputs from $checkpoint.."

            python generate.py \
                --data $DATA/tune.txt.json \
                --pred $checkpoint/qalb14_tune.preds.txt \
                --output $checkpoint/qalb14_tune.preds.gen.txt

        done

    done

done


