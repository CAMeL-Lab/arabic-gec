# This script is used to postprocess the generation models' outputs
# by comparing the output to the source. If the edits made to the source
# are greater than a certain thershold, then we just pass the source to
# the output. This is used to get around the M2 scorer efficiency issue.

experiment=qalb14
split=tune
INPUT_FILE=/scratch/ba63/gec/data/bart-t5/$experiment/wo_camelira/$split.json
OUTPUT_DIR=/scratch/ba63/gec/models/gec/${experiment}/bart_w_ged/ranking

for f in ${OUTPUT_DIR} # ${OUTPUT_DIR}/checkpoint*
    do
        printf "Postprocessing $f\n"
        python postprocess.py \
            --input $INPUT_FILE \
            --pred $f/qalb14_${split}.preds.ranked.txt \
            --output here # $f/qalb14_${split}.preds.pp.txt
        printf "\n\n"
    done

