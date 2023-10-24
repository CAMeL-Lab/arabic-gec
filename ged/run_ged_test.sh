#!/bin/bash
#SBATCH --reservation=v100_nlp
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

export DATASET=qalb14
export exp=qalb14
export DATA_DIR=/scratch/ba63/gec/data/ged++/${exp}/binary/wo_camelira
export OUTPUT_DIR=/scratch/ba63/gec/models/ged++/qalb14/binary/wo_camelira/checkpoint-500
export LABELS=/scratch/ba63/gec/data/ged++/qalb14/binary/wo_camelira/labels.txt
export BATCH_SIZE=32
export SEED=42
export pred_mode=dev


python error_detection.py \
     --data_dir $DATA_DIR \
     --labels $LABELS \
     --model_name_or_path $OUTPUT_DIR \
     --output_dir $OUTPUT_DIR \
     --per_device_eval_batch_size $BATCH_SIZE \
     --seed $SEED \
     --do_pred \
     --pred_output_file ${exp}_${pred_mode}.preds.txt.check \
     --pred_mode $pred_mode # or test to get the test predictions

     # Evaluation
        paste $DATA_DIR/${pred_mode}.txt $OUTPUT_DIR/${DATASET}_${pred_mode}.preds.txt.check \
            > $OUTPUT_DIR/eval_data_${pred_mode}_${DATASET}.txt.check

        python evaluate.py  --data $OUTPUT_DIR/eval_data_${pred_mode}_${DATASET}.txt.check \
                            --labels $LABELS \
                            --output $OUTPUT_DIR/${DATASET}_${pred_mode}.results.check

        rm $OUTPUT_DIR/eval_data_${pred_mode}_${DATASET}.txt.check

