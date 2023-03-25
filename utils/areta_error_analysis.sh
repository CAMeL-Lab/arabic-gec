#!/bin/bash
#SBATCH -q nlp
# Set number of tasks to run
#SBATCH --ntasks=1
# Set number of cores per task (default is 1)
#SBATCH --cpus-per-task=10 
# memory
#SBATCH --mem=120GB
# Walltime format hh:mm:ss
#SBATCH --time=47:59:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err

eval "$(conda shell.bash hook)"
conda activate areta

# SRC_DIR=/scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/dev/QALB-2015-L2-Dev.sent.no_ids.clean
# REF_DIR_FULL=/scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/dev/dev.areta
# REF_DIR_COARSE=/scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/dev/dev.areta.coarse
# REF_DIR_BINARY=/scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/dev/dev.areta.binary

SRC_DIR=/scratch/ba63/gec/data/ZAEBUC-v1.0/data/ar/dev/dev.sent.raw.pnx.tok
REF_DIR_FULL=/scratch/ba63/gec/data/ZAEBUC-v1.0/data/ar/dev/dev.areta
REF_DIR_COARSE=/scratch/ba63/gec/data/ZAEBUC-v1.0/data/ar/dev/dev.areta.coarse
REF_DIR_BINARY=/scratch/ba63/gec/data/ZAEBUC-v1.0/data/ar/dev/dev.areta.binary

EXP=dev.zaebuc.pred.non_oracle.txt.pnx.edits
EXP_oracle=dev.zaebuc.pred.txt.pnx.edits

PRED_ARETA_TAGS_FULL=/scratch/ba63/gec/ged-models/MIX/full_areta/dev_zaebuc_predictions.txt
GEN_DIR_FULL=/scratch/ba63/gec/models/MIX/t5_lr_with_pref_full_areta_30/checkpoint-65000
GEN_DIR_FULL_ORACLE=/scratch/ba63/gec/models/MIX/t5_lr_with_pref_full_areta_30

PRED_ARETA_TAGS_COARSE=/scratch/ba63/gec/ged-models/MIX/coarse_areta/dev_zaebuc_predictions.txt
GEN_DIR_COARSE=/scratch/ba63/gec/models/MIX/t5_lr_with_pref_coarse_areta_30/checkpoint-55000
GEN_DIR_COARSE_ORACLE=/scratch/ba63/gec/models/MIX/t5_lr_with_pref_coarse_areta_30

PRED_ARETA_TAGS_BINARY=/scratch/ba63/gec/ged-models/MIX/binary_areta/dev_zaebuc_predictions.txt
GEN_DIR_BINARY=/scratch/ba63/gec/models/MIX/t5_lr_with_pref_binary_areta_30
GEN_DIR_BINARY_ORACLE=/scratch/ba63/gec/models/MIX/t5_lr_with_pref_binary_areta_30/checkpoint-60000

OUTPUT_DIR=/home/ba63/gec/logs/error_analysis/ZAEBUC


for mode in FULL COARSE BINARY
do
    echo "Creating ARETA annotations between source and generated output ($mode).."
    
    # Adding s to the beginning of each src and tgt sentence to make areta happy
    pred_dir="GEN_DIR_${mode}"
    pred_dir_oracle="GEN_DIR_${mode}_ORACLE"

    cat $SRC_DIR | awk '{print "s "$0}' > $OUTPUT_DIR/src.areta
    cat ${!pred_dir}/$EXP | awk '{print "s "$0}' > $OUTPUT_DIR/pred.areta.$mode
    cat ${!pred_dir_oracle}/$EXP_oracle | awk '{print "s "$0}' > $OUTPUT_DIR/pred.areta.oracle.$mode

     python /home/ba63/arabic_error_type_annotation/annotate_err_type_ar.py \
         --sys_path $OUTPUT_DIR/src.areta \
         --ref_path $OUTPUT_DIR/pred.areta.$mode \
         --output_path $OUTPUT_DIR/pred.align.areta.$mode

    python /home/ba63/arabic_error_type_annotation/annotate_err_type_ar.py \
         --sys_path $OUTPUT_DIR/src.areta \
         --ref_path $OUTPUT_DIR/pred.areta.oracle.$mode \
         --output_path $OUTPUT_DIR/pred.align.areta.oracle.$mode

    printf "Finished generating ARETA annotations for ${mode}!\n\n"
done


printf "Generating Error Analysis..\n"
for mode in FULL COARSE BINARY
do
    
    # Getting source words and generated words, and the areta predicted GED tags
    cat $OUTPUT_DIR/pred.align.areta.$mode | cut -f1 | sed 's/^/<s>/' | sed 's/$/<s>/' > $OUTPUT_DIR/words_src.$mode.txt
    cat $OUTPUT_DIR/pred.align.areta.$mode | cut -f2 | sed 's/^/<s>/' | sed 's/$/<s>/' > $OUTPUT_DIR/words_gen.$mode.txt
    pred_areta_tags="PRED_ARETA_TAGS_${mode}"
    cat ${!pred_areta_tags} | cut -d' ' -f2 > $OUTPUT_DIR/src_pred_tags.$mode.txt

    #Getting the generated oracle words
    cat $OUTPUT_DIR/pred.align.areta.oracle.$mode | cut -f2 | sed 's/^/<s>/' | sed 's/$/<s>/' > $OUTPUT_DIR/words_gen.$mode.oracle.txt

    # Getting the source words and reference words, and the areta gold GED tags
    gold_areta="REF_DIR_${mode}"
    cat ${!gold_areta} | cut -f1 | sed 's/^/<s>/' | sed 's/$/<s>/' > $OUTPUT_DIR/words_src.check.txt
    cat ${!gold_areta} | cut -f2 | sed 's/^/<s>/' | sed 's/$/<s>/' > $OUTPUT_DIR/words_ref.txt
    cat ${!gold_areta} | cut -f3 > $OUTPUT_DIR/src_gold_tags.$mode.txt

done


# Checking that source words from prediction alignment and source words from gold data is the same
sdiff -s $OUTPUT_DIR/words_src.check.txt $OUTPUT_DIR/words_src.FULL.txt
sdiff -s $OUTPUT_DIR/words_src.check.txt $OUTPUT_DIR/words_src.COARSE.txt
sdiff -s $OUTPUT_DIR/words_src.check.txt $OUTPUT_DIR/words_src.BINARY.txt



echo -e 'SRC\tREF\tGOLD_TAG (FULL)\tGEN (FULL-ORACLE)\tGEN (FULL)\tPRED_TAG (FULL)\tGOLD_TAG' \
        '(COARSE)\tGEN (COARSE-ORACLE)\tGEN (COARSE)\tPRED_TAG (COARSE)\tGOLD_TAG (BINARY)'\
        '\tGEN (BINARY-ORACLE)\tGEN (BINARY)\tPRED_TAG (BINARY)' > $OUTPUT_DIR/error_analysis.tsv

paste $OUTPUT_DIR/words_src.FULL.txt $OUTPUT_DIR/words_ref.txt \
      $OUTPUT_DIR/src_gold_tags.FULL.txt $OUTPUT_DIR/words_gen.FULL.oracle.txt $OUTPUT_DIR/words_gen.FULL.txt \
      $OUTPUT_DIR/src_pred_tags.FULL.txt \
      $OUTPUT_DIR/src_gold_tags.COARSE.txt $OUTPUT_DIR/words_gen.COARSE.oracle.txt $OUTPUT_DIR/words_gen.COARSE.txt \
      $OUTPUT_DIR/src_pred_tags.COARSE.txt \
      $OUTPUT_DIR/src_gold_tags.BINARY.txt $OUTPUT_DIR/words_gen.BINARY.oracle.txt $OUTPUT_DIR/words_gen.BINARY.txt \
      $OUTPUT_DIR/src_pred_tags.BINARY.txt  >> $OUTPUT_DIR/error_analysis.tsv

rm $OUTPUT_DIR/words*.txt
rm $OUTPUT_DIR/*tags*.txt
rm $OUTPUT_DIR/pred.areta.*
rm $OUTPUT_DIR/src.areta
rm fout2.basic 