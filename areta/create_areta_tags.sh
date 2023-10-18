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
# with weird UNK issues..




# for data in qalb14 qalb15 zaebuc
# do
#     for split in train dev tune test L1-test L2-test
#     do

    # ALIGNMENT=/scratch/ba63/gec/data/alignment/modeling_alignment/${data}/${data}_${split}.txt
    ALIGNMENT=/scratch/ba63/gec/data/gec/segmented_data/qalb15/camelira_alignment/qalb15_segmented.test.alignment.txt

    if [ -f "$ALIGNMENT" ]; then
        printf "Creating ARETA Tags for $ALIGNMENT \n"

            # OUTPUT=/scratch/ba63/gec/data/alignment/modeling_areta_tags_improved/${data}/${data}_${split}.areta.txt
            # ENRICHED_OUTPUT=/scratch/ba63/gec/data/alignment/modeling_areta_tags_improved/${data}/${data}_${split}.areta+.txt
            OUTPUT=/scratch/ba63/gec/data/gec/segmented_data/qalb15/camelira_areta_tags/qalb15_segmented.test.areta.txt
            ENRICHED_OUTPUT=/scratch/ba63/gec/data/gec/segmented_data/qalb15/camelira_areta_tags/qalb15_segmented.test.areta+.txt


        python /home/ba63/arabic_error_type_annotation/annotate_err_type_ar.py \
            --alignment $ALIGNMENT \
            --output_path $OUTPUT \
            --enriched_output_path $ENRICHED_OUTPUT

        rm -r output
        rm fout2.basic

    else
        printf "$ALIGNMENT doesn't exist \n"
    fi

#     done
# done
