################ QALB-2014 ################

for split in train dev test 
do
    output_dir=/home/ba63/gec-release/data/ged/qalb14/wo_camelira/full
    areta_tags=/home/ba63/gec-release/data/areta_tags/wo_camelira/qalb14/qalb14_${split}.areta+.txt

    echo "Creating GED data using $areta_tags"

    python create_ged_data.py \
        --input $areta_tags \
        --output $output_dir/${split}.txt

    output_dir=/home/ba63/gec-release/data/ged/qalb14/w_camelira/full
    areta_tags=/home/ba63/gec-release/data/areta_tags/w_camelira/qalb14/qalb14_${split}.areta+.txt

    echo "Creating GED data using $areta_tags"

    python create_ged_data.py \
        --input $areta_tags \
        --output $output_dir/${split}.txt
done

################ QALB-2015  ################
output_dir=/home/ba63/gec-release/data/ged/qalb15/wo_camelira/full
areta_tags=home/ba63/gec-release/data/areta_tags/wo_camelira/qalb15

python create_ged_data.py \
    --input $areta_tags/qalb15_train.areta+.txt \
    --output $output_dir/train.txt

python create_ged_data.py \
    --input $areta_tags/qalb15_dev.areta+.txt \
    --output $output_dir/dev.txt

python create_ged_data.py \
     --input $areta_tags/qalb15_L2-test.areta+.txt \
    --output $output_dir/test_L2.txt


python create_ged_data.py \
     --input $areta_tags/qalb15_L1-test.areta+.txt \
    --output $output_dir/test_L1.txt


output_dir=/home/ba63/gec-release/data/ged/qalb15/w_camelira/full
areta_tags=home/ba63/gec-release/data/areta_tags/w_camelira/qalb15

python create_ged_data.py \
    --input $areta_tags/qalb15_train.areta+.txt \
    --output $output_dir/train.txt

python create_ged_data.py \
    --input $areta_tags/qalb15_dev.areta+.txt \
    --output $output_dir/dev.txt

python create_ged_data.py \
     --input $areta_tags/qalb15_L2-test.areta+.txt \
    --output $output_dir/test_L2.txt


python create_ged_data.py \
     --input $areta_tags/qalb15_L1-test.areta+.txt \
    --output $output_dir/test_L1.txt


################ ZAEBUC  ################
for split in train dev test
do
    output_dir=/home/ba63/gec-release/data/ged/zaebuc/wo_camelira/full
    areta_tags=home/ba63/gec-release/data/areta_tags/wo_camelira/zaebuc/zaebuc_${split}.areta+.txt

    echo "Creating GED data using $areta_tags"

    python create_ged_data.py \
        --input $areta_tags \
        --output $output_dir/${split}.txt

    output_dir=/home/ba63/gec-release/data/ged/zaebuc/w_camelira/full
    areta_tags=home/ba63/gec-release/data/areta_tags/w_camelira/zaebuc/zaebuc_${split}.areta+.txt

    echo "Creating GED data using $areta_tags"

    python create_ged_data.py \
        --input $areta_tags \
        --output $output_dir/${split}.txt
done


############### QALB-2014-2015 Train #############
output_dir=/home/ba63/gec-release/data/ged/qalb14-15/wo_camelira/full
qalb14=home/ba63/gec-release/data/areta_tags/wo_camelira/qalb14/qalb14_train.areta+.txt
qalb15=home/ba63/gec-release/data/areta_tags/wo_camelira/qalb15/qalb15_train.areta+.txt

{ cat ${qalb14}; sed '1d' ${qalb15}; }  > qalb14-15_train.areta+.txt

python create_ged_data.py \
     --input qalb14-15_train.areta+.txt \
     --output $output_dir/train.txt

rm qalb14-15_train.areta+.txt

output_dir=/home/ba63/gec-release/data/ged/qalb14-15/w_camelira/full
qalb14=home/ba63/gec-release/data/areta_tags/w_camelira/qalb14/qalb14_train.areta+.txt
qalb15=home/ba63/gec-release/data/areta_tags/w_camelira/qalb15/qalb15_train.areta+.txt

{ cat ${qalb14}; sed '1d' ${qalb15}; }  > qalb14-15_train.areta+.txt

python create_ged_data.py \
     --input qalb14-15_train.areta+.txt \
     --output $output_dir/train.txt

rm qalb14-15_train.areta+.txt


################ MIX Train ################
output_dir=/home/ba63/gec-release/data/ged/mix/wo_camelira/full
qalb14=home/ba63/gec-release/data/areta_tags/wo_camelira/qalb14/qalb14_train.areta+.txt
qalb15=home/ba63/gec-release/data/areta_tags/wo_camelira/qalb15/qalb15_train.areta+.txt
zaebuc=home/ba63/gec-release/data/areta_tags/wo_camelira/zaebuc/zaebuc_train.areta+.txt

{ cat ${qalb14}; sed '1d' ${qalb15};  sed '1d' ${zaebuc}; } > mix_train.areta+.txt

python create_ged_data.py \
     --input mix_train.areta+.txt \
     --output $output_dir/train.txt

rm mix_train.areta+.txt

output_dir=/home/ba63/gec-release/data/ged/mix/w_camelira/full
qalb14=home/ba63/gec-release/data/areta_tags/w_camelira/qalb14/qalb14_train.areta+.txt
qalb15=home/ba63/gec-release/data/areta_tags/w_camelira/qalb15/qalb15_train.areta+.txt
zaebuc=home/ba63/gec-release/data/areta_tags/w_camelira/zaebuc/zaebuc_train.areta+.txt

{ cat ${qalb14}; sed '1d' ${qalb15};  sed '1d' ${zaebuc}; } > mix_train.areta+.txt

python create_ged_data.py \
     --input mix_train.areta+.txt \
     --output $output_dir/train.txt

rm mix_train.areta+.txt