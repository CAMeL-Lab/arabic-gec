#!/bin/bash
output_dir=/home/ba63/gec-release/data/gec/modeling_check

#### QALB-2014 ####
qalb14_gec_dir=/home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014
qalb14_ged_dir=/home/ba63/gec-release/data/ged/qalb14/wo_camelira
qalb14_ged_pred_dir=/home/ba63/gec-release/ged/predictions/qalb14/wo_camelira

qalb14_gec_camelira_dir=/home/ba63/gec-release/data/gec/camelira_gec/qalb14
qalb14_ged_camelira_dir=/home/ba63/gec-release/data/ged/qalb14/w_camelira
qalb14_ged_pred_camelira_dir=/home/ba63/gec-release/ged/predictions/qalb14/w_camelira

### QALB-2014 ####

for split in train dev test
do
    for gran in full coarse binary
    do

        split_f="$(tr '[:lower:]' '[:upper:]' <<< ${split:0:1})${split:1}"

        paste ${qalb14_ged_dir}/${gran}/${split}.txt  ${qalb14_ged_pred_dir}/${gran}/qalb14_${split}.preds.txt | cut -f1,3 > tags.txt

        python jsonify_data.py  --src ${qalb14_gec_dir}/${split}/QALB-2014-L1-${split_f}.sent.no_ids.clean.dediac \
                                --tgt ${qalb14_gec_dir}/${split}/QALB-2014-L1-${split_f}.cor.no_ids.dediac \
                                --tags tags.txt \
                                --output $output_dir/qalb14/wo_camelira/${gran}/${split}.json

        python jsonify_data.py  --src ${qalb14_gec_dir}/${split}/QALB-2014-L1-${split_f}.sent.no_ids.clean.dediac \
                                --tgt ${qalb14_gec_dir}/${split}/QALB-2014-L1-${split_f}.cor.no_ids.dediac \
                                --tags ${qalb14_ged_dir}/${gran}/${split}.txt \
                                --output $output_dir/qalb14/wo_camelira/${gran}/${split}.oracle.json

        paste ${qalb14_ged_camelira_dir}/${gran}/${split}.txt  ${qalb14_ged_pred_camelira_dir}/${gran}/qalb14_${split}.preds.txt | cut -f1,3 > tags.txt

        split_f="$(tr '[:lower:]' '[:upper:]' <<< ${split:0:1})${split:1}"

        python jsonify_data.py  --src ${qalb14_gec_camelira_dir}/qalb14_${split}.src.txt  \
                                --tgt ${qalb14_gec_dir}/${split}/QALB-2014-L1-${split_f}.cor.no_ids.dediac \
                                --tags tags.txt \
                                --output $output_dir/qalb14/w_camelira/${gran}/${split}.json

        python jsonify_data.py  --src ${qalb14_gec_camelira_dir}/qalb14_${split}.src.txt  \
                                --tgt ${qalb14_gec_dir}/${split}/QALB-2014-L1-${split_f}.cor.no_ids.dediac \
                                --tags ${qalb14_ged_camelira_dir}/${gran}/${split}.txt \
                                --output $output_dir/qalb14/w_camelira/${gran}/${split}.oracle.json

        rm tags.txt
    done
done


for gran in full coarse binary
do

    paste ${qalb14_ged_dir}/${gran}/dev.txt  ${qalb14_ged_pred_dir}/${gran}/qalb14_qalb14-15_dev.preds.txt | cut -f1,3 > tags.txt

    python jsonify_data.py  --src ${qalb14_gec_dir}/dev/QALB-2014-L1-Dev.sent.no_ids.clean.dediac \
                            --tgt ${qalb14_gec_dir}/dev/QALB-2014-L1-Dev.cor.no_ids.dediac \
                            --tags tags.txt \
                            --output $output_dir/qalb14/wo_camelira/${gran}/dev_qalb14-15.json

    paste ${qalb14_ged_camelira_dir}/${gran}/dev.txt  ${qalb14_ged_pred_camelira_dir}/${gran}/qalb14_qalb14-15_dev.preds.txt | cut -f1,3 > tags.txt


    python jsonify_data.py  --src ${qalb14_gec_camelira_dir}/qalb14_dev.src.txt  \
                            --tgt ${qalb14_gec_dir}/dev/QALB-2014-L1-Dev.cor.no_ids.dediac \
                            --tags tags.txt \
                            --output $output_dir/qalb14/w_camelira/${gran}/dev_qalb14-15.json


    rm tags.txt
done


# ### QALB-2015 ####
qalb15_gec_dir=/home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015
qalb15_ged_dir=/home/ba63/gec-release/data/ged/qalb15/wo_camelira
qalb15_ged_pred_dir=/home/ba63/gec-release/ged/predictions/qalb15/wo_camelira


qalb15_gec_camelira_dir=/home/ba63/gec-release/data/gec/camelira_gec/qalb15
qalb15_ged_camelira_dir=/home/ba63/gec-release/data/ged/qalb15/w_camelira
qalb15_ged_pred_camelira_dir=/home/ba63/gec-release/ged/predictions/qalb15/w_camelira

for split in dev test_L1 test_L2
do
    for gran in full coarse binary
    do

        if [ "$split" = "test_L1" ]; then
            fname=QALB-2015-L1
            s="$(cut -d '_' -f1 <<< ${split})"
            lang=L1-

        elif [ "$split" = "test_L2" ]; then
            fname=QALB-2015-L2
            s="$(cut -d '_' -f1 <<< ${split})"
            lang=L2-
        else
            fname=QALB-2015-L2
            s=$split
            lang=""
        fi

        s_upp="$(tr '[:lower:]' '[:upper:]' <<< ${s:0:1})${s:1}"

        paste ${qalb15_ged_dir}/${gran}/${split}.txt  ${qalb15_ged_pred_dir}/${gran}/qalb15_${split}.preds.txt | cut -f1,3 > tags.txt

        python jsonify_data.py  --src ${qalb15_gec_dir}/${s}/${fname}-${s_upp}.sent.no_ids.dediac \
                                --tgt ${qalb15_gec_dir}/${s}/${fname}-${s_upp}.cor.no_ids.dediac \
                                --tags tags.txt \
                                --output $output_dir/qalb15/wo_camelira/${gran}/${split}.json

        python jsonify_data.py  --src ${qalb15_gec_dir}/${s}/${fname}-${s_upp}.sent.no_ids.dediac \
                                --tgt ${qalb15_gec_dir}/${s}/${fname}-${s_upp}.cor.no_ids.dediac \
                                --tags ${qalb15_ged_dir}/${gran}/${split}.txt \
                                --output $output_dir/qalb15/wo_camelira/${gran}/${split}.oracle.json

        paste ${qalb15_ged_camelira_dir}/${gran}/${split}.txt  ${qalb15_ged_pred_camelira_dir}/${gran}/qalb15_${split}.preds.txt | cut -f1,3 > tags.txt

        python jsonify_data.py  --src ${qalb15_gec_camelira_dir}/qalb15_${lang}${s}.src.txt \
                                --tgt ${qalb15_gec_dir}/${s}/${fname}-${s_upp}.cor.no_ids.dediac \
                                --tags tags.txt \
                                --output $output_dir/qalb15/w_camelira/${gran}/${split}.json

        python jsonify_data.py  --src ${qalb15_gec_camelira_dir}/qalb15_${lang}${s}.src.txt \
                                --tgt ${qalb15_gec_dir}/${s}/${fname}-${s_upp}.cor.no_ids.dediac \
                                --tags ${qalb15_ged_camelira_dir}/${gran}/${split}.txt \
                                --output $output_dir/qalb15/w_camelira/${gran}/${split}.oracle.json

        rm tags.txt
    done
done


# ### ZAEBUC ####
zaebuc_gec_dir=/home/ba63/gec-release/data/gec/ZAEBUC-v1.0/data/ar
zaebuc_ged_dir=/home/ba63/gec-release/data/ged/zaebuc/wo_camelira
zaebuc_ged_pred_dir=/home/ba63/gec-release/ged/predictions/zaebuc/wo_camelira

zaebuc_gec_camelira_dir=/home/ba63/gec-release/data/gec/camelira_gec/zaebuc
zaebuc_ged_camelira_dir=/home/ba63/gec-release/data/ged/zaebuc/w_camelira
zaebuc_ged_pred_camelira_dir=/home/ba63/gec-release/ged/predictions/zaebuc/w_camelira

for split in dev test
do
    for gran in full coarse binary
    do

        split_f="$(tr '[:lower:]' '[:upper:]' <<< ${split:0:1})${split:1}"

        paste ${zaebuc_ged_dir}/${gran}/${split}.txt  ${zaebuc_ged_pred_dir}/${gran}/zaebuc_${split}.preds.txt | cut -f1,3 > tags.txt

        python jsonify_data.py  --src ${zaebuc_gec_dir}/${split}/${split}.sent.raw.pnx.tok.dediac \
                                --tgt ${zaebuc_gec_dir}/${split}/${split}.sent.cor.pnx.tok.dediac \
                                --tags tags.txt \
                                --output $output_dir/zaebuc/wo_camelira/${gran}/${split}.json

        python jsonify_data.py  --src ${zaebuc_gec_dir}/${split}/${split}.sent.raw.pnx.tok.dediac \
                                --tgt ${zaebuc_gec_dir}/${split}/${split}.sent.cor.pnx.tok.dediac \
                                --tags ${zaebuc_ged_dir}/${gran}/${split}.txt \
                                --output $output_dir/zaebuc/wo_camelira/${gran}/${split}.oracle.json

        paste ${zaebuc_ged_camelira_dir}/${gran}/${split}.txt  ${zaebuc_ged_pred_camelira_dir}/${gran}/zaebuc_${split}.preds.txt | cut -f1,3 > tags.txt

        python jsonify_data.py  --src ${zaebuc_gec_camelira_dir}/zaebuc_${split}.src.txt \
                                --tgt ${zaebuc_gec_dir}/${split}/${split}.sent.cor.pnx.tok.dediac \
                                --tags tags.txt \
                                --output $output_dir/zaebuc/w_camelira/${gran}/${split}.json

        python jsonify_data.py  --src ${zaebuc_gec_camelira_dir}/zaebuc_${split}.src.txt \
                                --tgt ${zaebuc_gec_dir}/${split}/${split}.sent.cor.pnx.tok.dediac \
                                --tags ${zaebuc_ged_camelira_dir}/${gran}/${split}.txt \
                                --output $output_dir/zaebuc/w_camelira/${gran}/${split}.oracle.json

        rm tags.txt

    done
done


## QALB14-15 Train ####

qalb14_15_gec_dir=/home/ba63/gec-release/data/gec/qalb14-15
qalb14_15_ged_dir=/home/ba63/gec-release/data/ged/qalb14-15/wo_camelira
qalb14_15_ged_pred_dir=/home/ba63/gec-release/ged/predictions/qalb14-15/wo_camelira

qalb14_15_gec_camelira_dir=/home/ba63/gec-release/data/gec/camelira_gec/qalb14-15
qalb14_15_ged_camelira_dir=/home/ba63/gec-release/data/ged/qalb14-15/w_camelira
qalb14_15_ged_pred_camelira_dir=/home/ba63/gec-release/ged/predictions/qalb14-15/w_camelira

for gran in full coarse binary
do
    paste ${qalb14_15_ged_dir}/${gran}/train.txt  ${qalb14_15_ged_pred_dir}/${gran}/qalb14-15_train.preds.txt | cut -f1,3 > tags.txt

    python jsonify_data.py  --src ${qalb14_15_gec_dir}/train.sent.dediac \
                            --tgt ${qalb14_15_gec_dir}/train.cor.dediac \
                            --tags tags.txt \
                            --output $output_dir/qalb14-15/wo_camelira/${gran}/train.json

    python jsonify_data.py  --src ${qalb14_15_gec_dir}/train.sent.dediac \
                            --tgt ${qalb14_15_gec_dir}/train.cor.dediac \
                            --tags ${qalb14_15_ged_dir}/${gran}/train.txt \
                            --output $output_dir/qalb14-15/wo_camelira/${gran}/train.oracle.json

    paste ${qalb14_15_ged_camelira_dir}/${gran}/train.txt  ${qalb14_15_ged_pred_camelira_dir}/${gran}/qalb14-15_train.preds.txt | cut -f1,3 > tags.txt

    python jsonify_data.py  --src ${qalb14_15_gec_camelira_dir}/train.src.txt \
                            --tgt ${qalb14_15_gec_dir}/train.cor.dediac \
                            --tags tags.txt \
                            --output $output_dir/qalb14-15/w_camelira/${gran}/train.json

    python jsonify_data.py  --src ${qalb14_15_gec_camelira_dir}/train.src.txt \
                            --tgt ${qalb14_15_gec_dir}/train.cor.dediac \
                            --tags ${qalb14_15_ged_camelira_dir}/${gran}/train.txt \
                            --output $output_dir/qalb14-15/w_camelira/${gran}/train.oracle.json

    rm tags.txt
done


### MIX Train ####
mix_gec_dir=/home/ba63/gec-release/data/gec/mix
mix_ged_dir=/home/ba63/gec-release/data/ged/mix/wo_camelira
mix_ged_pred_dir=/home/ba63/gec-release/ged/predictions/mix/wo_camelira

mix_gec_camelira_dir=/home/ba63/gec-release/data/gec/camelira_gec/mix
mix_ged_camelira_dir=/home/ba63/gec-release/data/ged/mix/w_camelira
mix_ged_pred_camelira_dir=/home/ba63/gec-release/ged/predictions/mix/w_camelira

for gran in full coarse binary
do
    paste ${mix_ged_dir}/${gran}/train.txt  ${mix_ged_pred_dir}/${gran}/mix_train.preds.txt | cut -f1,3 > tags.txt

    python jsonify_data.py  --src ${mix_gec_dir}/train.sent.dediac \
                            --tgt ${mix_gec_dir}/train.cor.dediac \
                            --tags tags.txt \
                            --output $output_dir/mix/wo_camelira/${gran}/train.json

    python jsonify_data.py  --src ${mix_gec_dir}/train.sent.dediac \
                            --tgt ${mix_gec_dir}/train.cor.dediac \
                            --tags ${mix_ged_dir}/${gran}/train.txt \
                            --output $output_dir/mix/wo_camelira/${gran}/train.oracle.json

    paste ${mix_ged_camelira_dir}/${gran}/train.txt  ${mix_ged_pred_camelira_dir}/${gran}/mix_train.preds.txt | cut -f1,3 > tags.txt

    python jsonify_data.py  --src ${mix_gec_camelira_dir}/train.src.txt \
                            --tgt ${mix_gec_dir}/train.cor.dediac \
                            --tags tags.txt \
                            --output $output_dir/mix/w_camelira/${gran}/train.json

    python jsonify_data.py  --src ${mix_gec_camelira_dir}/train.src.txt \
                            --tgt ${mix_gec_dir}/train.cor.dediac \
                            --tags ${mix_ged_camelira_dir}/${gran}/train.txt \
                            --output $output_dir/mix/w_camelira/${gran}/train.oracle.json

    rm tags.txt
done