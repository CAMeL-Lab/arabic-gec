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

export DATASET=qalb15
export exp=qalb15
export DATA_DIR=/scratch/ba63/gec/data/ged++/${exp}/coarse/w_camelira
export OUTPUT_DIR=/scratch/ba63/gec/models/ged++/qalb14-15/coarse/w_camelira/checkpoint-2000
export LABELS=/scratch/ba63/gec/data/ged++/qalb14-15/coarse/w_camelira/labels.txt
export BATCH_SIZE=32
export SEED=42
export pred_mode=test_L1



python error_detection.py \
     --data_dir $DATA_DIR \
     --labels $LABELS \
     --model_name_or_path $OUTPUT_DIR \
     --output_dir $OUTPUT_DIR \
     --per_device_eval_batch_size $BATCH_SIZE \
     --seed $SEED \
     --do_pred \
     --pred_output_file ${exp}_${pred_mode}.preds.txt \
     --pred_mode $pred_mode # or test to get the test predictions

     # Evaluation
        paste $DATA_DIR/${pred_mode}.txt $OUTPUT_DIR/${DATASET}_${pred_mode}.preds.txt \
            > $OUTPUT_DIR/eval_data_${pred_mode}_${DATASET}.txt

        python evaluate_new.py --data $OUTPUT_DIR/eval_data_${pred_mode}_${DATASET}.txt \
                               --labels $LABELS \
                               --output $OUTPUT_DIR/${DATASET}_${pred_mode}.results

        rm $OUTPUT_DIR/eval_data_${pred_mode}_${DATASET}.txt

