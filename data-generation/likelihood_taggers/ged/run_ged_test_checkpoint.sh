#!/bin/bash
# SBATCH -q nlp
#SBATCH -p nvidia
# use gpus
#SBATCH --gres=gpu:1
# memory
#SBATCH --mem=120000
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
# --labels $DATA_DIR/areta.labels.${GRANULARITY}.txt \
# --labels /scratch/ba63/gec/ged-data/MIX/areta.labels.${GRANULARITY}.txt \

export DATA_DIR=/scratch/ba63/gec/ged-data/qalb14/full
export MAX_LENGTH=512
export OUTPUT_DIR=/scratch/ba63/gec/ged-models/qalb14/full/multi-class/model_check
export BATCH_SIZE=32
export SEED=42

for checkpoint in $OUTPUT_DIR/checkpoint-* $OUTPUT_DIR
# for checkpoint in $OUTPUT_DIR
do

    # cp $OUTPUT_DIR/tokenizer_config.json $checkpoint
    # cp $OUTPUT_DIR/vocab.txt $checkpoint
    # cp $OUTPUT_DIR/special_tokens_map.json $checkpoint

    printf "Running evaluation using ${checkpoint}..\n"

    # python error_detection.py \
    # --data_dir $DATA_DIR \
    # --labels $DATA_DIR/multi-class/labels.txt \
    # --model_name_or_path $checkpoint \
    # --output_dir $checkpoint \
    # --max_seq_length  $MAX_LENGTH \
    # --per_device_eval_batch_size $BATCH_SIZE \
    # --seed $SEED \
    # --do_pred \
    # --pred_output_file tune_mix_predictions.txt \
    # --pred_mode tune # or test to get the test predictions

    # computing metrics
    python metrics.py --pred $checkpoint/tune_qalb14_predictions.cascaded.txt \
                      --gold $DATA_DIR/tune_w_labels.txt \
                      --output $checkpoint/tune_qalb14_predictions.cascaded.metrics \
                      --labels $DATA_DIR/multi-label/labels.txt 

done
