#!/bin/bash
#SBATCH -p nvidia
#SBATCH -q nlp
#SBATCH --reservation=nlp-gpu
# use gpus
#SBATCH --gres=gpu:v100:1
# memory
#SBATCH --mem=120GB
# Walltime format hh:mm:ss
#SBATCH --time=47:59:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err

nvidia-smi
module purge

#################################
# ERROR DETECTION TEST EVAL SCRIPT
#################################
export exp=qalb15
export DATA_DIR=/scratch/ba63/gec/data/ged++/${exp}/w_camelira
export OUTPUT_DIR=/scratch/ba63/gec/models/ged++/qalb14-15/w_camelira
export LABELS=/scratch/ba63/gec/data/ged++/qalb14-15/w_camelira/labels.txt
export BATCH_SIZE=32
export SEED=42
export pred_mode=dev


for checkpoint in $OUTPUT_DIR/checkpoint-* $OUTPUT_DIR 

do
    # cp $OUTPUT_DIR/tokenizer_config.json $checkpoint
    # cp $OUTPUT_DIR/vocab.txt $checkpoint
    # cp $OUTPUT_DIR/special_tokens_map.json $checkpoint

    printf "Running evaluation using ${checkpoint}..\n"

    python error_detection.py \
        --data_dir $DATA_DIR \
        --labels $LABELS \
        --model_name_or_path $checkpoint \
        --output_dir $checkpoint \
        --per_device_eval_batch_size $BATCH_SIZE \
        --seed $SEED \
        --do_pred \
        --pred_output_file ${exp}_${pred_mode}.preds.txt \
        --pred_mode $pred_mode # or test to get the test predictions


    python metrics.py --pred $checkpoint/${exp}_${pred_mode}.preds.txt \
                      --gold $DATA_DIR/${pred_mode}.txt \
                      --labels $DATA_DIR/single_labels.txt \
                      --include_uc \
                      --output $checkpoint/${exp}_${pred_mode}.results.txt


done
