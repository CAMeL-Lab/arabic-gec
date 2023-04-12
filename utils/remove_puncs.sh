#!/bin/bash


input=zaebuc_dev.preds.merge_fix.txt

out1=/scratch/ba63/gec/models/gec/mix/t5_w_ged
out2=/scratch/ba63/gec/models/gec/mix/t5_w_camelira_ged
out3=/scratch/ba63/gec/models/gec/mix/t5_w_ged_pred_worst
out4=/scratch/ba63/gec/models/gec/mix/t5_w_camelira_ged_pred_worst

# out3=/scratch/ba63/gec/models/gec/qalb14/t5_w_ged_pred_random
# out4=/scratch/ba63/gec/models/gec/qalb14/t5_w_camelira_ged

for f in   ${out1} ${out1}/checkpoint-* ${out2} ${out2}/checkpoint-* ${out3} ${out3}/checkpoint-* ${out4} ${out4}/checkpoint-*
    do
        printf "Removing pnx from $f/$input\n"
        python remove_puncs.py \
            --input $f/$input \
            --output $f/$input.nopnx
    done