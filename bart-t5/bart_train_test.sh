MODEL=/scratch/ba63/BERT_models/AraBART
# MODEL=/scratch/ba63/BERT_models/AraT5-base
# MODEL=/scratch/ba63/BERT_models/AraT5-msa-base
# --source_prefix "convert raw to cor: " \
# --validation_file /scratch/ba63/gec/bart-t5-data/ZAEBUC/dev.json

OUTPUT_DIR=/scratch/ba63/gec/models/QALB-2014/bart-test

python run_gec.py \
    --model_name_or_path $MODEL \
    --do_train \
    --source_lang raw \
    --target_lang cor \
    --save_steps 500 \
    --train_file /scratch/ba63/gec/bart-t5-data/QALB-2014/train.areta.binary.json \
    --areta_tags  /scratch/ba63/gec/bart-t5-data/QALB-2014/areta.labels.binary.txt \
    --remove_unused_columns False \
    --num_train_epochs 10 \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --max_target_length 1024 \
    --seed 42 \
    --overwrite_cache \
    --overwrite_output_dir
