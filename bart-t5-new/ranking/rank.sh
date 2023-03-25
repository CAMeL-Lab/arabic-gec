
model_dir=/scratch/ba63/gec/models/gec/qalb14/bart_w_ged

for checkpoint in $model_dir $model_dir/checkpoint*
do
    printf "Ranking the outputs for $checkpoint..\n"
    python rank.py \
        --src_w_ged /scratch/ba63/gec/data/bart-t5/qalb14/wo_camelira/tune.json \
        --model_dir $checkpoint

done
