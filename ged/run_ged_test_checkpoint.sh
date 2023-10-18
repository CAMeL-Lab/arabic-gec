#!/bin/bash
#SBATCH -p nvidia
# use gpus
#SBATCH --gres=gpu:v100:1
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

DATASET=qalb14
BATCH_SIZE=32
SEED=42
pred_mode=dev

for GRANULARITY in  full coarse binary
do

    DATA_DIR=/scratch/ba63/gec/data/ged++/${DATASET}/${GRANULARITY}/w_camelira
    OUTPUT_DIR=/scratch/ba63/gec/models/ged++/qalb14-15/${GRANULARITY}/w_camelira
    LABELS=/scratch/ba63/gec/data/ged++/qalb14-15/${GRANULARITY}/w_camelira/labels.txt

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
            --pred_output_file ${DATASET}_${pred_mode}.preds.txt \
            --pred_mode $pred_mode # or test to get the test predictions


        # Evaluation
        paste $DATA_DIR/${pred_mode}.txt $checkpoint/${DATASET}_${pred_mode}.preds.txt \
            > $checkpoint/eval_data_${pred_mode}_${DATASET}.txt

        python evaluate_new.py --data $checkpoint/eval_data_${pred_mode}_${DATASET}.txt \
                               --labels $LABELS \
                               --output $checkpoint/${DATASET}_${pred_mode}.results

        rm $checkpoint/eval_data_${pred_mode}_${DATASET}.txt

    done

done
