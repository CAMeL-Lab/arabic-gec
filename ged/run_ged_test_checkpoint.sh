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

export DATA_DIR=/scratch/ba63/gec/data/ged/qalb14/w_camelira
export MAX_LENGTH=256
export OUTPUT_DIR=/scratch/ba63/gec/models/ged/qalb14/w_camelira
export BATCH_SIZE=32
export SEED=42
export pred_mode=tune

for checkpoint in $OUTPUT_DIR/checkpoint-* $OUTPUT_DIR
# for checkpoint in $OUTPUT_DIR
do
    # cp $OUTPUT_DIR/sentencepiece.bpe.model $checkpoint
    # cp $OUTPUT_DIR/tokenizer_config.json $checkpoint
    # cp $OUTPUT_DIR/vocab.txt $checkpoint
    # cp $OUTPUT_DIR/special_tokens_map.json $checkpoint

    printf "Running evaluation using ${checkpoint}..\n"

    # python error_detection.py \
    #     --data_dir $DATA_DIR \
    #     --labels $DATA_DIR/labels.txt \
    #     --model_name_or_path $checkpoint \
    #     --output_dir $checkpoint \
    #     --max_seq_length  $MAX_LENGTH \
    #     --per_device_eval_batch_size $BATCH_SIZE \
    #     --seed $SEED \
    #     --do_pred \
    #     --pred_output_file qalb14_${pred_mode}.preds.txt \
    #     --pred_mode $pred_mode # or test to get the test predictions

    # computing metrics
    python metrics.py --pred $checkpoint/qalb14_${pred_mode}.preds.txt \
                      --gold $DATA_DIR/tune_w_labels.txt \
                      --labels $DATA_DIR/single_labels.txt \
                      --output $checkpoint/qalb14_${pred_mode}.results.no_uc.txt

done
