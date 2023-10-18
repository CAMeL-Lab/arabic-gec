#!/bin/bash
#SBATCH -p nvidia
#SBATCH -q nlp
# use gpus
#SBATCH --gres=gpu:1
# memory
#SBATCH --mem=120000
# Walltime format hh:mm:ss
#SBATCH --time=47:59:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err

dir=/scratch/ba63/gec/data/gec/segmented_data/qalb15

for split in train dev test
do

python gec_by_camelbert.py \
    --input_file $dir/src_tgt/qalb15_segmented.$split.src.txt  \
    --output_file $dir/camelira_src/qalb15_segmented.$split.src.txt

done

# for split in train dev tune test
# do
#     split_f="$(tr '[:lower:]' '[:upper:]' <<< ${split:0:1})${split:1}"

#     SRC=/scratch/ba63/gec/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/$split/QALB-2014-L1-${split_f}.sent.no_ids.clean.dediac.nopnx
#     OUT=/scratch/ba63/gec/data/gec_camelira/qalb14/qalb14_$split.src.txt.nopnx

#     printf "Correcting $SRC\n"
#     python gec_by_camelbert.py \
#         --input_file $SRC \
#         --output_file $OUT
# done


# for split in train dev test
# do
#     split_f="$(tr '[:lower:]' '[:upper:]' <<< ${split:0:1})${split:1}"

#     SRC=/scratch/ba63/gec/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/$split/QALB-2015-L2-${split_f}.sent.no_ids.dediac.nopnx
#     OUT=/scratch/ba63/gec/data/gec_camelira/qalb15/qalb15_$split.src.txt.nopnx

#     printf "Correcting $SRC\n"

#     python gec_by_camelbert.py \
#         --input_file $SRC \
#         --output_file $OUT
# done

# SRC=/scratch/ba63/gec/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/test/QALB-2015-L1-Test.sent.no_ids.dediac.nopnx
# OUT=/scratch/ba63/gec/data/gec_camelira/qalb15/qalb15_L1-test.src.txt.nopnx

# printf "Correcting $SRC\n"

# python gec_by_camelbert.py \
#     --input_file $SRC \
#     --output_file $OUT



# for split in train dev test
# do
#     split_f="$(tr '[:lower:]' '[:upper:]' <<< ${split:0:1})${split:1}"

#     SRC=/scratch/ba63/gec/data/gec/ZAEBUC-v1.0/data/ar/$split/$split.sent.raw.pnx.tok.dediac.nopnx
#     OUT=/scratch/ba63/gec/data/gec_camelira/zaebuc/zaebuc_$split.src.txt.nopnx

#     printf "Correcting $SRC\n"

#     python gec_by_camelbert.py \
#         --input_file $SRC \
#         --output_file $OUT

# done
