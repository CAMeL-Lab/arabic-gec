#!/bin/bash

qalb14_dir=gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014
qalb15_dir=gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015
zaebuc_dir=gec/ZAEBUC-v1.0/data/ar

# QALB-2014 preprocessing
for split in train dev test
do
    s="$(tr '[:lower:]' '[:upper:]' <<< ${split:0:1})${split:1}"

    printf "#### QALB-2014 ${split} ####\n"

    # removing the ids from the files
    cat ${qalb14_dir}/$split/QALB-2014-L1-${s}.sent | cut -d' ' -f2-  > ${qalb14_dir}/$split/QALB-2014-L1-${s}.sent.no_ids
    cat ${qalb14_dir}/$split/QALB-2014-L1-${s}.cor | cut -d' ' -f2-  > ${qalb14_dir}/$split/QALB-2014-L1-${s}.cor.no_ids

    # QALB 2014 has an issue with quranic verses on the src
    python preprocess_qalb14.py \
        --input_file  ${qalb14_dir}/$split/QALB-2014-L1-${s}.sent.no_ids \
        --output_file ${qalb14_dir}/$split/QALB-2014-L1-${s}.sent.no_ids.clean

    # dediac the src
    python dediac.py \
        --input_file ${qalb14_dir}/$split/QALB-2014-L1-${s}.sent.no_ids.clean \
        --output_file ${qalb14_dir}/$split/QALB-2014-L1-${s}.sent.no_ids.clean.dediac

    # dediac the tgt
    python dediac.py \
        --input_file ${qalb14_dir}/$split/QALB-2014-L1-${s}.cor.no_ids \
        --output_file ${qalb14_dir}/$split/QALB-2014-L1-${s}.cor.no_ids.dediac

    # removing pnx from src
    python remove_puncs.py \
        --input_file ${qalb14_dir}/$split/QALB-2014-L1-${s}.sent.no_ids \
        --output_file ${qalb14_dir}/$split/QALB-2014-L1-${s}.sent.no_ids.nopnx


    # removing pnx from tgt
    python remove_puncs.py \
        --input_file ${qalb14_dir}/$split/QALB-2014-L1-${s}.cor.no_ids \
        --output_file ${qalb14_dir}/$split/QALB-2014-L1-${s}.cor.no_ids.nopnx

    # removing pnx from src dediac
    python remove_puncs.py \
        --input_file ${qalb14_dir}/$split/QALB-2014-L1-${s}.sent.no_ids.clean.dediac \
        --output_file ${qalb14_dir}/$split/QALB-2014-L1-${s}.sent.no_ids.clean.dediac.nopnx


    # removing pnx from tgt dediac
    python remove_puncs.py \
        --input_file ${qalb14_dir}/$split/QALB-2014-L1-${s}.cor.no_ids.dediac \
        --output_file ${qalb14_dir}/$split/QALB-2014-L1-${s}.cor.no_ids.dediac.nopnx


done


# QALB-2015 preprocessing
for split in train dev test-L1 test-L2
do
    printf "\n#### QALB-2015 ${split} ####\n"

    if [ "$split" = "test-L1" ]; then
        fname=QALB-2015-L1
        split="$(cut -d '-' -f1 <<< ${split})"
    elif [ "$split" = "test-L2" ]; then
        fname=QALB-2015-L2
        split="$(cut -d '-' -f1 <<< ${split})"
    else
        fname=QALB-2015-L2
    fi

    s="$(tr '[:lower:]' '[:upper:]' <<< ${split:0:1})${split:1}"

    # removing the ids from the files
    cat ${qalb15_dir}/$split/${fname}-${s}.sent | cut -d' ' -f2-  > ${qalb15_dir}/$split/${fname}-${s}.sent.no_ids
    cat ${qalb15_dir}/$split/${fname}-${s}.cor | cut -d' ' -f2-  > ${qalb15_dir}/$split/${fname}-${s}.cor.no_ids

    # dediac the src
    python dediac.py \
        --input_file ${qalb15_dir}/$split/${fname}-${s}.sent.no_ids \
        --output_file ${qalb15_dir}/$split/${fname}-${s}.sent.no_ids.dediac

    # dediac the tgt
    python dediac.py \
        --input_file ${qalb15_dir}/$split/${fname}-${s}.cor.no_ids \
        --output_file ${qalb15_dir}/$split/${fname}-${s}.cor.no_ids.dediac

    # removing pnx from src 
    python remove_puncs.py \
        --input_file ${qalb15_dir}/$split/${fname}-${s}.sent.no_ids \
        --output_file ${qalb15_dir}/$split/${fname}-${s}.sent.no_ids.nopnx


    # removing pnx from tgt 
    python remove_puncs.py \
        --input_file ${qalb15_dir}/$split/${fname}-${s}.cor.no_ids \
        --output_file ${qalb15_dir}/$split/${fname}-${s}.cor.no_ids.nopnx


    # removing pnx from src dediac
    python remove_puncs.py \
        --input_file ${qalb15_dir}/$split/${fname}-${s}.sent.no_ids.dediac \
        --output_file ${qalb15_dir}/$split/${fname}-${s}.sent.no_ids.dediac.nopnx


    # removing pnx from tgt dediac
    python remove_puncs.py \
        --input_file ${qalb15_dir}/$split/${fname}-${s}.cor.no_ids.dediac \
        --output_file ${qalb15_dir}/$split/${fname}-${s}.cor.no_ids.dediac.nopnx

done


# ZAEBUC preporcessing
for split in train dev test
do
    printf "\n#### ZAEBUC ${split} ####\n"

    # dediac the src
    python dediac.py \
        --input_file  ${zaebuc_dir}/${split}/${split}.sent.raw.pnx.tok \
        --output_file ${zaebuc_dir}/${split}/${split}.sent.raw.pnx.tok.dediac

    # dediac the tgt
    python dediac.py \
        --input_file ${zaebuc_dir}/${split}/${split}.sent.cor.pnx.tok \
        --output_file ${zaebuc_dir}/${split}/${split}.sent.cor.pnx.tok.dediac

    # removing pnx from src
    python remove_puncs.py \
        --input_file ${zaebuc_dir}/${split}/${split}.sent.raw.pnx.tok \
        --output_file ${zaebuc_dir}/${split}/${split}.sent.raw.pnx.tok.nopnx


    # removing pnx from tgt
    python remove_puncs.py \
        --input_file ${zaebuc_dir}/${split}/${split}.sent.cor.pnx.tok \
        --output_file ${zaebuc_dir}/${split}/${split}.sent.cor.pnx.tok.nopnx


    # removing pnx from src dediac
    python remove_puncs.py \
        --input_file ${zaebuc_dir}/${split}/${split}.sent.raw.pnx.tok.dediac \
        --output_file ${zaebuc_dir}/${split}/${split}.sent.raw.pnx.tok.dediac.nopnx


    # removing pnx from tgt dediac
    python remove_puncs.py \
        --input_file ${zaebuc_dir}/${split}/${split}.sent.cor.pnx.tok.dediac \
        --output_file ${zaebuc_dir}/${split}/${split}.sent.cor.pnx.tok.dediac.nopnx

done
