#!/bin/bash
#SBATCH -p nvidia
#SBATCH -q nlp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120GB
#SBATCH --time=47:59:00
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err

# CREATE THE ALIGNMENT WE NEED FOR M2 Files

qalb14_dir=/home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014
qalb15_dir=/home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015
zaebuc_dir=/home/ba63/gec-release/data/gec/ZAEBUC-v1.0/data/ar

qalb14_dir_camelira=/home/ba63/gec-release/data/gec/camelira_gec/qalb14
qalb15_dir_camelira=/home/ba63/gec-release/data/gec/camelira_gec/qalb15
zaebuc_dir_camelira=/home/ba63/gec-release/data/gec/camelira_gec/zaebuc

m2_alignments_dir=/home/ba63/gec-release/data/alignments/m2
modeling_alignments_dir=/home/ba63/gec-release/data/alignments/modeling
modeling_alignments_camelira_dir=/home/ba63/gec-release/data/alignments/modeling_camelira


if [ "$1" = "m2" ]; then
    printf "Creating alignments for m2 edits...\n"

    ### QALB-2014 ####
    for split in dev test
    do
        s="$(tr '[:lower:]' '[:upper:]' <<< ${split:0:1})${split:1}"

        python /home/ba63/gec-release/alignment/aligner.py \
                --src ${qalb14_dir}/${split}/QALB-2014-L1-${s}.sent.no_ids \
                --tgt ${qalb14_dir}/${split}/QALB-2014-L1-${s}.cor.no_ids \
                --output ${m2_alignments_dir}/qalb14/qalb14_${split}.txt

        python /home/ba63/gec-release/alignment/aligner.py \
                --src ${qalb14_dir}/${split}/QALB-2014-L1-${s}.sent.no_ids.nopnx \
                --tgt ${qalb14_dir}/${split}/QALB-2014-L1-${s}.cor.no_ids.nopnx \
                --output ${m2_alignments_dir}/qalb14/qalb14_${split}.nopnx.txt
    done

    #### QALB-2015 ####
    for split in dev L1-test L2-test
    do
        if [ "$split" = "L1-test" ]; then
            fname=QALB-2015-L1
            lang=L1-
            split="$(cut -d '-' -f2 <<< ${split})"

        elif [ "$split" = "L2-test" ]; then
                fname=QALB-2015-L2
                split="$(cut -d '-' -f2 <<< ${split})"
                lang=L2-
        else
                fname=QALB-2015-L2
                lang=""
        fi

        s="$(tr '[:lower:]' '[:upper:]' <<< ${split:0:1})${split:1}"

        python /home/ba63/gec-release/alignment/aligner.py \
                --src ${qalb15_dir}/${split}/${fname}-${s}.sent.no_ids \
                --tgt ${qalb15_dir}/${split}/${fname}-${s}.cor.no_ids \
                --output ${m2_alignments_dir}/qalb15/qalb15_${lang}${split}.txt

        python /home/ba63/gec-release/alignment/aligner.py \
                --src ${qalb15_dir}/${split}/${fname}-${s}.sent.no_ids.nopnx \
                --tgt ${qalb15_dir}/${split}/${fname}-${s}.cor.no_ids.nopnx \
                --output ${m2_alignments_dir}/qalb15/qalb15_${lang}${split}.nopnx.txt
    done

    #### ZAEBUC ####
    for split in dev test
    do
        s="$(tr '[:lower:]' '[:upper:]' <<< ${split:0:1})${split:1}"

        python /home/ba63/gec-release/alignment/aligner.py \
                --src ${zaebuc_dir}/${split}/${split}.sent.raw.pnx.tok \
                --tgt ${zaebuc_dir}/${split}/${split}.sent.cor.pnx.tok \
                --output ${m2_alignments_dir}/zaebuc/zaebuc_${split}.txt

        python /home/ba63/gec-release/alignment/aligner.py \
                --src ${zaebuc_dir}/${split}/${split}.sent.raw.pnx.tok.nopnx \
                --tgt ${zaebuc_dir}/${split}/${split}.sent.cor.pnx.tok.nopnx \
                --output ${m2_alignments_dir}/zaebuc/zaebuc_${split}.nopnx.txt
    done




# CREATE THE ALIGNMENT WE NEED FOR MODELING (i.e., removing diacs and kashidas)
elif [ "$1" = "modeling" ]; then
    printf "Creating alignments for modeling...\n"

    #### QALB-2014 ####
    for split in train dev test
    do
        s="$(tr '[:lower:]' '[:upper:]' <<< ${split:0:1})${split:1}"

        python /home/ba63/gec-release/alignment/aligner.py \
                --src ${qalb14_dir}/${split}/QALB-2014-L1-${s}.sent.no_ids.clean.dediac \
                --tgt ${qalb14_dir}/${split}/QALB-2014-L1-${s}.cor.no_ids.dediac \
                --output ${modeling_alignments_dir}/qalb14/qalb14_${split}.txt

        python /home/ba63/gec-release/alignment/aligner.py \
                --src ${qalb14_dir}/${split}/QALB-2014-L1-${s}.sent.no_ids.clean.dediac.nopnx \
                --tgt ${qalb14_dir}/${split}/QALB-2014-L1-${s}.cor.no_ids.dediac.nopnx \
                --output ${modeling_alignments_dir}/qalb14/qalb14_${split}.nopnx.txt

        # CAMELIRA data
        python /home/ba63/gec-release/alignment/aligner.py \
                --src ${qalb14_dir_camelira}/qalb14_${split}.src.txt \
                --tgt ${qalb14_dir}/${split}/QALB-2014-L1-${s}.cor.no_ids.dediac \
                --output ${modeling_alignments_camelira_dir}/qalb14/qalb14_${split}.txt

        python /home/ba63/gec-release/alignment/aligner.py \
                --src ${qalb14_dir_camelira}/qalb14_${split}.src.txt.nopnx \
                --tgt ${qalb14_dir}/${split}/QALB-2014-L1-${s}.cor.no_ids.dediac.nopnx \
                --output ${modeling_alignments_camelira_dir}/qalb14/qalb14_${split}.nopnx.txt

    done

    #### QALB-2015 ####
    for split in train dev L1-test L2-test
    do
        if [ "$split" = "L1-test" ]; then
            fname=QALB-2015-L1
            lang=L1-
            split="$(cut -d '-' -f2 <<< ${split})"

        elif [ "$split" = "L2-test" ]; then
                fname=QALB-2015-L2
                split="$(cut -d '-' -f2 <<< ${split})"
                lang=L2-
        else
                fname=QALB-2015-L2
                lang=""
        fi

        s="$(tr '[:lower:]' '[:upper:]' <<< ${split:0:1})${split:1}"

        python /home/ba63/gec-release/alignment/aligner.py \
                --src ${qalb15_dir}/${split}/${fname}-${s}.sent.no_ids.dediac \
                --tgt ${qalb15_dir}/${split}/${fname}-${s}.cor.no_ids.dediac \
                --output ${modeling_alignments_dir}/qalb15/qalb15_${lang}${split}.txt

        python /home/ba63/gec-release/alignment/aligner.py \
                --src ${qalb15_dir}/${split}/${fname}-${s}.sent.no_ids.dediac.nopnx \
                --tgt ${qalb15_dir}/${split}/${fname}-${s}.cor.no_ids.dediac.nopnx \
                --output ${modeling_alignments_dir}/qalb15/qalb15_${lang}${split}.nopnx.txt

        # CAMELIRA data
        python /home/ba63/gec-release/alignment/aligner.py \
                --src ${qalb15_dir_camelira}/qalb15_${lang}${split}.src.txt \
                --tgt ${qalb15_dir}/${split}/${fname}-${s}.cor.no_ids.dediac \
                --output ${modeling_alignments_camelira_dir}/qalb15/qalb15_${lang}${split}.txt

        python /home/ba63/gec-release/alignment/aligner.py \
                --src ${qalb15_dir_camelira}/qalb15_${lang}${split}.src.txt.nopnx \
                --tgt ${qalb15_dir}/${split}/${fname}-${s}.cor.no_ids.dediac.nopnx \
                --output ${modeling_alignments_camelira_dir}/qalb15/qalb15_${lang}${split}.nopnx.txt

    done

    #### ZAEBUC ####
    for split in train dev test
    do
        s="$(tr '[:lower:]' '[:upper:]' <<< ${split:0:1})${split:1}"

        python /home/ba63/gec-release/alignment/aligner.py \
                --src ${zaebuc_dir}/${split}/${split}.sent.raw.pnx.tok.dediac \
                --tgt ${zaebuc_dir}/${split}/${split}.sent.cor.pnx.tok.dediac \
                --output ${modeling_alignments_dir}/zaebuc/zaebuc_${split}.txt

        python /home/ba63/gec-release/alignment/aligner.py \
                --src ${zaebuc_dir}/${split}/${split}.sent.raw.pnx.tok.dediac.nopnx \
                --tgt ${zaebuc_dir}/${split}/${split}.sent.cor.pnx.tok.dediac.nopnx \
                --output ${modeling_alignments_dir}/zaebuc/zaebuc_${split}.nopnx.txt

        # CAMELIRA data
        python /home/ba63/gec-release/alignment/aligner.py \
                --src ${zaebuc_dir_camelira}/zaebuc_${split}.src.txt \
                --tgt ${zaebuc_dir}/${split}/${split}.sent.cor.pnx.tok.dediac \
                --output ${modeling_alignments_camelira_dir}/zaebuc/zaebuc_${split}.txt

        python /home/ba63/gec-release/alignment/aligner.py \
                --src ${zaebuc_dir_camelira}/zaebuc_${split}.src.txt.nopnx \
                --tgt ${zaebuc_dir}/${split}/${split}.sent.cor.pnx.tok.dediac.nopnx \
                --output ${modeling_alignments_camelira_dir}/zaebuc/zaebuc_${split}.nopnx.txt

    done

fi
