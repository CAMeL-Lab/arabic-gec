# ############# MIX UP SAMPLE ################
OUTPUT_DIR=/scratch/ba63/gec/data/ged++/mix_up/w_camelira
ALIGN_DIR=/scratch/ba63/gec/data/gec_camelira/areta_tags/mix_up

python create_ged_data.py \
    --input $ALIGN_DIR/train.areta+.txt \
    --output $OUTPUT_DIR/train.txt


########## QALB-2014-2015 UP SAMPLE #############
OUTPUT_DIR=/scratch/ba63/gec/data/ged++/qalb14-15_up/w_camelira
ALIGN_DIR=/scratch/ba63/gec/data/gec_camelira/areta_tags/qalb14-15_up

python create_ged_data.py \
     --input $ALIGN_DIR/train.areta+.txt \
     --output $OUTPUT_DIR/train.txt




################ MIX ################
# OUTPUT_DIR=/scratch/ba63/gec/data/ged++/mix/w_camelira
# ALIGN_DIR=/scratch/ba63/gec/data/gec_camelira/areta_tags/mix

# python create_ged_data.py \
#     --input $ALIGN_DIR/train.areta+.txt \
#     --output $OUTPUT_DIR/train.txt


############### QALB-2014-2015 #############
# OUTPUT_DIR=/scratch/ba63/gec/data/ged++/qalb14-15/w_camelira
# ALIGN_DIR=/scratch/ba63/gec/data/gec_camelira/areta_tags/qalb14-15

# python create_ged_data.py \
#      --input $ALIGN_DIR/train.areta+.txt \
#      --output $OUTPUT_DIR/train.txt



################ QALB-2014  ################
# OUTPUT_DIR=/scratch/ba63/gec/data/ged++/qalb14/w_camelira
# ALIGN_DIR=/scratch/ba63/gec/data/gec_camelira/areta_tags/qalb14


# for split in train tune dev test 
# do
#     split_f="$(tr '[:lower:]' '[:upper:]' <<< ${split:0:1})${split:1}"

#     align_file=$ALIGN_DIR/qalb14_${split}.areta+.txt

#     echo "Creating GED data using $align_file"

#     python create_ged_data.py \
#         --input $align_file \
#         --output $OUTPUT_DIR/${split}.txt

# done


################ QALB-2015  ################
# OUTPUT_DIR=/scratch/ba63/gec/data/ged++/qalb15/w_camelira
# ALIGN_DIR=/scratch/ba63/gec/data/gec_camelira/areta_tags/qalb15

# for split in train dev
# do
#     split_f="$(tr '[:lower:]' '[:upper:]' <<< ${split:0:1})${split:1}"

#     align_file=$ALIGN_DIR/qalb15_${split}.areta+.txt

#     echo "Creating GED data using $align_file"

#     python create_ged_data.py \
#         --input $align_file \
#         --output $OUTPUT_DIR/${split}.txt
    
# done


# align_file=$ALIGN_DIR/qalb15_L2-test.areta+.txt
# python create_ged_data.py \
#     --input $align_file \
#     --output $OUTPUT_DIR/test_L2.txt

# align_file=$ALIGN_DIR/qalb15_L1-test.areta+.txt
# python create_ged_data.py \
#     --input $align_file \
#     --output $OUTPUT_DIR/test_L1.txt


################ ZAEBUC  ################
# OUTPUT_DIR=/scratch/ba63/gec/data/ged++/zaebuc/w_camelira
# ALIGN_DIR=/scratch/ba63/gec/data/gec_camelira/areta_tags/zaebuc


# for split in train dev test
# do
#     align_file=$ALIGN_DIR/zaebuc_${split}.areta+.txt

#     echo "Creating GED data using $align_file"

#     python create_ged_data.py \
#         --input $align_file \
#         --output $OUTPUT_DIR/${split}.txt


# done
