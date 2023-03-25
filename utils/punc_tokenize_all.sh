#!/bin/bash

# INPUT_FILE=/scratch/ba63/gec/bart-t5-data/MIX/dev.json
# OUTPUT_DIR=/scratch/ba63/gec/models/MIX/rules_tagger/3_pruned

# for f in ${OUTPUT_DIR} ${OUTPUT_DIR}/checkpoint*
#     do
#         printf "Postprocessing $f\n"
#         python postprocess.py --input_file_dir $INPUT_FILE \
#             --pred_file_dir $f/predicted_iter2.txt \
#             --output_file_dir $f/predicted_iter2.txt.pnx.edits
#         printf "\n\n"
#     done


OUTPUT_DIR=/scratch/ba63/gec/models/zaebuc/bart

for f in ${OUTPUT_DIR} ${OUTPUT_DIR}/checkpoint*
    do
        printf "Punctuation tokenizing $f\n"
        python punc_tokenize.py \
            --input $OUTPUT_DIR/zaebuc_dev_pred.txt \
            --output $OUTPUT_DIR/zaebuc_dev_pred.txt.punc_tokenized
        printf "\n\n"
        mv $OUTPUT_DIR/zaebuc_dev_pred.txt.punc_tokenized $OUTPUT_DIR/zaebuc_dev_pred.txt
    done
