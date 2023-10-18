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



model_dir=/scratch/ba63/gec/models/gec/qalb14/t5_w_ged

for checkpoint in $model_dir/ranking $model_dir/checkpoint-*/ranking
do
    for hyp in {1..5}
    do
        align_file=$checkpoint/src.to.${hyp}.alignment

        printf "Creating ARETA Tags for $align_file \n"

        python /home/ba63/arabic_error_type_annotation/annotate_err_type_ar.py \
            --alignment $align_file \
            --output_path $checkpoint/src.to.${hyp}.areta.txt \
            --enriched_output_path $checkpoint/src.to.${hyp}.areta+.txt

        rm fout2.basic

    done
done
