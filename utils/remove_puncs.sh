#!/bin/bash

INPUT_FILE=qalb14_tune.preds.gen.txt
out1=/scratch/ba63/gec/models/rules_tagger/wo_camelira/all
out2=/scratch/ba63/gec/models/rules_tagger/wo_camelira/4
out3=/scratch/ba63/gec/models/rules_tagger/wo_camelira/5
out4=/scratch/ba63/gec/models/rules_tagger/wo_camelira/10

for f in ${out1} ${out1}/checkpoint-* ${out2} ${out2}/checkpoint-* ${out3} ${out3}/checkpoint-* ${out4} ${out4}/checkpoint-*
    do
        printf "Removing pnx from $f/$INPUT_FILE\n"
        python remove_puncs.py \
            --input $f/$INPUT_FILE \
            --output $f/$INPUT_FILE.nopnx
    done



