#!/bin/bash
#SBATCH -p nvidia
#SBATCH -q nlp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120GB
#SBATCH --time=47:59:00
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err

 eval "$(conda shell.bash hook)"
 conda activate areta


# Important note: we modified the areta code and now it expects an alignment
# file rather than a source and a target. We should modify areta 
# to take a source and a target and to create the alignment internally
# Another important note: this bash script should be ran from the areta directory to deal 
# with weird UNK issues. Make sure to run this as an interactive job!

cd /home/ba63/gec-release/areta

for data in qalb14 qalb15 zaebuc
do
    for split in train dev test L1-test L2-test
    do

    alignment=/home/ba63/gec-release/data/alignments/modeling/${data}/${data}_${split}.txt

    if [ -f "$alignment" ]; then
        printf "Creating ARETA Tags for $alignment \n"

        output=/home/ba63/gec-release/data/areta_tags/wo_camelira/${data}/${data}_${split}.areta.txt
        enriched_output=/home/ba63/gec-release/data/areta_tags/wo_camelira/${data}/${data}_${split}.areta+.txt

        python /home/ba63/gec-release/areta/annotate_err_type_ar.py \
            --alignment $alignment \
            --output_path $output \
            --enriched_output_path $enriched_output

        rm fout2.basic
        rm /home/ba63/gec-release/data/alignments/modeling/${data}/*tsv

    else
        printf "$alignment doesn't exist \n"
    fi

    alignment_camelira=/home/ba63/gec-release/data/alignments/modeling_camelira/${data}/${data}_${split}.txt

    if [ -f "$alignment_camelira" ]; then
        printf "Creating ARETA Tags for $alignment_camelira \n"

        output=/home/ba63/gec-release/data/areta_tags/w_camelira/${data}/${data}_${split}.areta.txt
        enriched_output=/home/ba63/gec-release/data/areta_tags/w_camelira/${data}/${data}_${split}.areta+.txt

        python /home/ba63/gec-release/areta/annotate_err_type_ar.py \
            --alignment $alignment_camelira \
            --output_path $output \
            --enriched_output_path $enriched_output

        rm fout2.basic
        rm /home/ba63/gec-release/data/alignments/modeling_camelira/${data}/*tsv

    else
        printf "$alignment_camelira doesn't exist \n"
    fi

    done
done


### NO PNX areta tags ###

for data in qalb14 qalb15 zaebuc
do
    for split in train dev test L1-test L2-test
    do

    alignment_nopnx=/home/ba63/gec-release/data/alignments/modeling/${data}/${data}_${split}.nopnx.txt

    if [ -f "$alignment_nopnx" ]; then
        printf "Creating ARETA Tags for $alignment_nopnx \n"

        output=/home/ba63/gec-release/data/areta_tags/wo_camelira/${data}/${data}_${split}.areta.nopnx.txt
        enriched_output=/home/ba63/gec-release/data/areta_tags/wo_camelira/${data}/${data}_${split}.areta+.nopnx.txt

        python /home/ba63/gec-release/areta/annotate_err_type_ar.py \
            --alignment $alignment_nopnx \
            --output_path $output \
            --enriched_output_path $enriched_output

        rm fout2.basic
        rm /home/ba63/gec-release/data/alignments/modeling/${data}/*tsv

    else
        printf "$alignment_nopnx doesn't exist \n"
    fi

    alignment_camelira_nopnx=/home/ba63/gec-release/data/alignments/modeling_camelira/${data}/${data}_${split}.nopnx.txt

    if [ -f "$alignment_camelira_nopnx" ]; then
        printf "Creating ARETA Tags for $alignment_camelira_nopnx \n"

        output=/home/ba63/gec-release/data/areta_tags/w_camelira/${data}/${data}_${split}.areta.nopnx.txt
        enriched_output=/home/ba63/gec-release/data/areta_tags/w_camelira/${data}/${data}_${split}.areta+.nopnx.txt

        python /home/ba63/gec-release/areta/annotate_err_type_ar.py \
            --alignment $alignment_camelira_nopnx \
            --output_path $output \
            --enriched_output_path $enriched_output

        rm fout2.basic
        rm /home/ba63/gec-release/data/alignments/modeling_camelira/${data}/*tsv

    else
        printf "$alignment_camelira_nopnx doesn't exist \n"
    fi

    done
done
