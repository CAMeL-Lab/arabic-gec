# This script is used to postprocess the generation models' outputs
# by comparing the output to the source. If the edits made to the source
# are greater than a certain thershold, then we just pass the source to
# the output. This is used to get around the M2 scorer efficiency issue.

experiment=mix
split=tune
INPUT_FILE=/scratch/ba63/gec/data/bart-t5/$experiment/w_camelira/$split.json
OUTPUT_DIR=/scratch/ba63/gec/models/gec/mix/bart_w_camelira_ged/checkpoint-1000

for f in ${OUTPUT_DIR} # ${OUTPUT_DIR}/checkpoint*
    do
        printf "Postprocessing $f\n"
        python postprocess.py \
            --input $INPUT_FILE \
            --pred $f/mix_${split}.preds.merge_fix.txt \
            --output here
        printf "\n\n"
    done

