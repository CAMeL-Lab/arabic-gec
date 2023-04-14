#!/usr/bin/env bash
# areta_tags=""

# DATA_DIR=/scratch/ba63/gec/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks
# DATA_DIR=/scratch/ba63/gec/data/gec/qalb14-15
# DATA_DIR=/scratch/ba63/gec/data/gec/mix
DATA_DIR=/scratch/ba63/gec/data/gec/ZAEBUC-v1.0/data/ar
CAMELIRA_SRC=/scratch/ba63/gec/data/gec_camelira/zaebuc
# OUTPUT_DIR=/scratch/ba63/gec/data/bart-t5/qalb14/w_camelira
# OUTPUT_DIR=/scratch/ba63/gec/data/bart-t5/qalb14-15/w_camelira
# OUTPUT_DIR=/scratch/ba63/gec/data/bart-t5/mix/w_camelira
# OUTPUT_DIR=/scratch/ba63/gec/data/bart-t5/qalb15/w_camelira
OUTPUT_DIR=/scratch/ba63/gec/data/bart-t5/zaebuc/wo_camelira

python jsonify_data.py  --src $DATA_DIR/dev/dev.sent.raw.pnx.tok.dediac \
                        --tgt $DATA_DIR/dev/dev.sent.cor.pnx.tok.dediac \
                        --tags $OUTPUT_DIR/dev_mix_preds.txt \
                        --output $OUTPUT_DIR/check.json



# python jsonify_data.py  --src $CAMELIRA_SRC/qalb14/qalb14_dev.src.txt \
#                         --tgt $DATA_DIR/data/2014/dev/QALB-2014-L1-Dev.cor.no_ids.dediac \
#                         --tags /scratch/ba63/gec/data/ged++/qalb14/w_camelira/dev.txt \
#                         --output $OUTPUT_DIR/dev.json


# DATA_DIR=/scratch/ba63/gec/data/gec/ZAEBUC-v1.0/data/ar/dev

# OUTPUT_DIR=/scratch/ba63/gec/data/bart-t5/qalb15/wo_camelira
# OUTPUT_DIR=/scratch/ba63/gec/data/bart-t5/zaebuc/wo_camelira


# python jsonify_data.py  --src /scratch/ba63/gec/data/gec_camelira/mix/mix_train.src.txt \
#                         --tgt $DATA_DIR/train.cor.dediac \
#                         --tags /scratch/ba63/gec/data/ged++/mix/w_camelira/train.txt \
#                         --output $OUTPUT_DIR/train.json

# python jsonify_data.py  --src $DATA_DIR/dev.sent.raw.pnx.tok.dediac \
#                         --tgt $DATA_DIR/dev.sent.cor.pnx.tok.dediac \
#                         --tags $OUTPUT_DIR/dev_preds.txt \
#                         --output $OUTPUT_DIR/dev_preds.check.json






# python jsonify_data.py  --src $DATA_DIR/data/2014/train/QALB-2014-L1-Train.sent.no_ids.clean.dediac \
#                         --tgt $DATA_DIR/data/2014/train/QALB-2014-L1-Train.cor.no_ids.dediac \
#                         --tags $OUTPUT_DIR/train_preds.txt \
#                         --output $OUTPUT_DIR/train_preds.json


# python jsonify_data.py  --src $DATA_DIR/data/2014/train/QALB-2014-L1-Train.sent.no_ids.clean.dediac \
#                         --tgt $DATA_DIR/data/2014/train/QALB-2014-L1-Train.cor.no_ids.dediac \
#                         --tags /scratch/ba63/gec/data/ged/qalb14/wo_camelira/train.txt \
#                         --output $OUTPUT_DIR/train.json

# python jsonify_data.py  --src $DATA_DIR/data/2014/tune/QALB-2014-L1-Tune.sent.no_ids.clean.dediac \
#                         --tgt $DATA_DIR/data/2014/tune/QALB-2014-L1-Tune.cor.no_ids.dediac \
#                         --tags /scratch/ba63/gec/data/ged/qalb14/wo_camelira/tune_w_labels.txt \
#                         --output $OUTPUT_DIR/tune.json

# python jsonify_data.py  --src $DATA_DIR/data/2014/tune/QALB-2014-L1-Tune.sent.no_ids.clean.dediac \
#                         --tgt $DATA_DIR/data/2014/tune/QALB-2014-L1-Tune.cor.no_ids.dediac \
#                         --tags /scratch/ba63/gec/data/bart-t5/qalb14/wo_camelira/cheeck \
#                        --output $OUTPUT_DIR/tune_preds_check.json


# python jsonify_data.py   --src /scratch/ba63/gec/data/gec_camelira/qalb14/qalb14_train.src.txt \
#                          --tgt $DATA_DIR/data/2014/train/QALB-2014-L1-Train.cor.no_ids.dediac \
#                          --tags /scratch/ba63/gec/data/ged/qalb14/w_camelira/train.txt \
#                          --output $OUTPUT_DIR/train.json


# python jsonify_data.py   --src /scratch/ba63/gec/data/gec_camelira/qalb14/qalb14_tune.src.txt \
#                          --tgt $DATA_DIR/data/2014/tune/QALB-2014-L1-Tune.cor.no_ids.dediac \
#                          --tags /scratch/ba63/gec/data/ged/qalb14/w_camelira/tune_w_labels.txt \
#                          --output $OUTPUT_DIR/tune.json


# python jsonify_data.py   --src /scratch/ba63/gec/data/gec_camelira/qalb14/qalb14_tune.src.txt \
#                          --tgt $DATA_DIR/data/2014/tune/QALB-2014-L1-Tune.cor.no_ids.dediac \
#                         --tags $OUTPUT_DIR/tune_preds.txt \
#                         --output $OUTPUT_DIR/tune_preds.json


#  python jsonify_data.py   --src /scratch/ba63/gec/data/gec_camelira/qalb14/qalb14_train.src.txt \
#                           --tgt $DATA_DIR/data/2014/train/QALB-2014-L1-Train.cor.no_ids.dediac \
#                           --tags $OUTPUT_DIR/checkme \
#                           --output $OUTPUT_DIR/train_preds_worst.json


# python jsonify_data.py  --src $DATA_DIR/data/2014/train/QALB-2014-L1-Train.sent.no_ids.clean.dediac \
#                         --tgt $DATA_DIR/data/2014/train/QALB-2014-L1-Train.cor.no_ids.dediac \
#                         --tags $OUTPUT_DIR/train_preds.random.txt \
#                         --output $OUTPUT_DIR/train_preds.random.json
