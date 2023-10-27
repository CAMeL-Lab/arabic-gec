#!/bin/bash
#SBATCH --reservation=v100_nlp
#SBATCH -p nvidia
# use gpus
#SBATCH --gres=gpu:v100:1
# memory
#SBATCH --mem=120000
# Walltime format hh:mm:ss
#SBATCH --time=47:59:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err


qalb14_dir=/home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014
qalb15_dir=/home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015
zaebuc_dir=/home/ba63/gec-release/data/gec/ZAEBUC-v1.0/data/ar


qalb14_out_dir=/home/ba63/gec-release/data/gec/camelira_gec/qalb14
qalb15_out_dir=/home/ba63/gec-release/data/gec/camelira_gec/qalb15
zaebuc_out_dir=/home/ba63/gec-release/data/gec/camelira_gec/zaebuc

#### QALB-2014 ####
for split in train dev test
do
    split_f="$(tr '[:lower:]' '[:upper:]' <<< ${split:0:1})${split:1}"

    src=${qalb14_dir}/$split/QALB-2014-L1-${split_f}.sent.no_ids.clean.dediac
    out=${qalb14_out_dir}/qalb14_$split.src.txt

    printf "Correcting $src\n"
    python camelira_gec.py \
        --input_file $src \
        --output_file $out

    src=${qalb14_dir}/$split/QALB-2014-L1-${split_f}.sent.no_ids.clean.dediac.nopnx
    out=${qalb14_out_dir}/qalb14_$split.src.txt.nopnx

    printf "Correcting $src\n"
    python camelira_gec.py \
        --input_file $src \
        --output_file $out

done

#### QALB-2015 ####
for split in train dev test
do
    split_f="$(tr '[:lower:]' '[:upper:]' <<< ${split:0:1})${split:1}"
    src=${qalb15_dir}/${split}/QALB-2015-L2-${split_f}.sent.no_ids.dediac

    if [ "$split" = "test" ]; then
        out=${qalb15_out_dir}/qalb15_L2-test.src.txt
    else
        out=${qalb15_out_dir}/qalb15_$split.src.txt
    fi

    printf "Correcting $src\n"

    python camelira_gec.py \
        --input_file $src \
        --output_file $out


    src=${qalb15_dir}/${split}/QALB-2015-L2-${split_f}.sent.no_ids.dediac.nopnx

    if [ "$split" = "test" ]; then
        out=${qalb15_out_dir}/qalb15_L2-test.src.txt.nopnx
    else
        out=${qalb15_out_dir}/qalb15_$split.src.txt.nopnx
    fi

    printf "Correcting $src\n"

    python camelira_gec.py \
        --input_file $src \
        --output_file $out
done



src=${qalb15_dir}/test/QALB-2015-L1-Test.sent.no_ids.dediac
out=${qalb15_out_dir}/qalb15_L1-test.src.txt

printf "Correcting $src\n"

python camelira_gec.py \
    --input_file $src \
    --output_file $out

src=${qalb15_dir}/test/QALB-2015-L1-Test.sent.no_ids.dediac.nopnx
out=${qalb15_out_dir}/qalb15_L1-test.src.txt.nopnx

printf "Correcting $src\n"

python camelira_gec.py \
    --input_file $src \
    --output_file $out


#### ZAEBUC ####
for split in train dev test
do
    split_f="$(tr '[:lower:]' '[:upper:]' <<< ${split:0:1})${split:1}"

    src=${zaebuc_dir}/$split/$split.sent.raw.pnx.tok.dediac
    out=${zaebuc_out_dir}/zaebuc_$split.src.txt

    printf "Correcting $src\n"

    python camelira_gec.py \
        --input_file $src \
        --output_file $out


    src=${zaebuc_dir}/$split/$split.sent.raw.pnx.tok.dediac.nopnx
    out=${zaebuc_out_dir}/zaebuc_$split.src.txt.nopnx

    printf "Correcting $src\n"

    python camelira_gec.py \
        --input_file $src \
        --output_file $out
done
