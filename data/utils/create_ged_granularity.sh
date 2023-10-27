data_dir=/home/ba63/gec-release/data/ged/qalb14

for split in train dev test
do
    python granularity_map.py \
        --input $data_dir/wo_camelira/full/${split}.txt \
        --mode binary \
        --output $data_dir/wo_camelira/binary/${split}.txt

    python granularity_map.py \
        --input $data_dir/wo_camelira/full/${split}.txt \
        --mode coarse \
        --output $data_dir/wo_camelira/coarse/${split}.txt

    python granularity_map.py \
        --input $data_dir/w_camelira/full/${split}.txt \
        --mode binary \
        --output $data_dir/w_camelira/binary/${split}.txt

    python granularity_map.py \
        --input $data_dir/w_camelira/full/${split}.txt \
        --mode coarse \
        --output $data_dir/w_camelira/coarse/${split}.txt


done


data_dir=/home/ba63/gec-release/data/ged/qalb15

for split in dev test_L1 test_L2
do
    python granularity_map.py \
        --input $data_dir/wo_camelira/full/${split}.txt \
        --mode binary \
        --output $data_dir/wo_camelira/binary/${split}.txt

    python granularity_map.py \
        --input $data_dir/wo_camelira/full/${split}.txt \
        --mode coarse \
        --output $data_dir/wo_camelira/coarse/${split}.txt

    python granularity_map.py \
        --input $data_dir/w_camelira/full/${split}.txt \
        --mode binary \
        --output $data_dir/w_camelira/binary/${split}.txt

    python granularity_map.py \
        --input $data_dir/w_camelira/full/${split}.txt \
        --mode coarse \
        --output $data_dir/w_camelira/coarse/${split}.txt
done


data_dir=/home/ba63/gec-release/data/ged/zaebuc

for split in dev test
do
    python granularity_map.py \
        --input $data_dir/wo_camelira/full/${split}.txt \
        --mode binary \
        --output $data_dir/wo_camelira/binary/${split}.txt

    python granularity_map.py \
        --input $data_dir/wo_camelira/full/${split}.txt \
        --mode coarse \
        --output $data_dir/wo_camelira/coarse/${split}.txt

    python granularity_map.py \
        --input $data_dir/w_camelira/full/${split}.txt \
        --mode binary \
        --output $data_dir/w_camelira/binary/${split}.txt

    python granularity_map.py \
        --input $data_dir/w_camelira/full/${split}.txt \
        --mode coarse \
        --output $data_dir/w_camelira/coarse/${split}.txt
done


for data_dir in /home/ba63/gec-release/data/ged/qalb14-15 /home/ba63/gec-release/data/ged/mix
do
    python granularity_map.py \
        --input $data_dir/wo_camelira/full/train.txt \
        --mode binary \
        --output $data_dir/wo_camelira/binary/train.txt

    python granularity_map.py \
        --input $data_dir/wo_camelira/full/train.txt \
        --mode coarse \
        --output $data_dir/wo_camelira/coarse/train.txt

    python granularity_map.py \
        --input $data_dir/w_camelira/full/train.txt \
        --mode binary \
        --output $data_dir/w_camelira/binary/train.txt

    python granularity_map.py \
        --input $data_dir/w_camelira/full/train.txt \
        --mode coarse \
        --output $data_dir/w_camelira/coarse/train.txt

done