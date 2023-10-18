# model=/scratch/ba63/gec/ged-models/multi-class/model_new_latest_fix_reg_new
# DATA_DIR=/scratch/ba63/gec/data/alignment/modeling_areta_tags/qalb14/multi-class

# for checkpoint in $model $model/checkpoint-*
# do
#     python metrics.py --pred_path $checkpoint/tune_qalb14_predictions.txt \
#                       --gold_path $DATA_DIR/tune_w_labels.txt \
#                       --output_path $checkpoint/tune_qalb14_predictions.metrics \
#                       --labels_path /scratch/ba63/gec/data/alignment/modeling_areta_tags/qalb14/multi-label/labels.txt

# done


# model=/scratch/ba63/gec/models/ged++/qalb14-15/wo_camelira
# DATA_DIR=/scratch/ba63/gec/data/ged++/qalb14/full/wo_camelira

# for checkpoint in $model $model/checkpoint-*
# do
#     paste $DATA_DIR/dev.txt $checkpoint/qalb14_dev.preds.txt \
#         | cut -f1,3 > $checkpoint/qalb14_dev.preds.eval.txt

# done


model=/scratch/ba63/gec/models/ged++/qalb14-15/wo_camelira
ref=/scratch/ba63/gec/data/ged++/qalb14/full/wo_camelira/dev.txt

for checkpoint in $model $model/checkpoint-*
do
    python eval.py \
        -hyp $checkpoint/qalb14_dev.preds.eval.txt \
        -ref $ref > $checkpoint/qalb14.eval.new

done


