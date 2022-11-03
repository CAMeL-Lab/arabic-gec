#!/bin/bash

INPUT_FILE=/scratch/ba63/gec/bart-t5-data/ZAEBUC/dev.json
OUTPUT_DIR=/scratch/ba63/gec/models/QALB-2014/t5_lr_with_pref_30

for f in ${OUTPUT_DIR} ${OUTPUT_DIR}/checkpoint-*
    do
        printf "Postprocessing $f\n"
        python postprocess.py --input_file_dir $INPUT_FILE --pred_file_dir $f/dev.zaebuc.pred.txt --output_file_dir $f/dev.zaebuc.pred.txt.pnx.edits
        printf "\n\n"
    done
