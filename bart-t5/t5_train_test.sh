# MODEL=/scratch/ba63/BERT_models/AraBART
MODEL=/scratch/ba63/BERT_models/AraT5-base
# MODEL=/scratch/ba63/BERT_models/AraT5-msa-base
# --source_prefix "convert raw to cor: " \
# --validation_file /scratch/ba63/gec/bart-t5-data/ZAEBUC/dev.json

OUTPUT_DIR=/scratch/ba63/gec/models/QALB-2014/t5_lr_with_pref_30_check

python run_gec.py \
    --model_name_or_path $MODEL \
    --do_train \
    --source_lang raw \
    --target_lang cor \
    --source_prefix "convert raw to cor: " \
    --train_file /scratch/ba63/gec/bart-t5-data/QALB-2014/train.areta.binary.json \
    --areta_tags  /scratch/ba63/gec/bart-t5-data/QALB-2014/areta.labels.binary.txt \
    --save_steps 1500 \
    --num_train_epochs 30 \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --max_target_length 1024 \
    --seed 42 \
    --learning_rate 1e-04 \
    --overwrite_cache \
    --overwrite_output_dir
