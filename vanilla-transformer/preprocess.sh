#!/bin/bash
#SBATCH -p nvidia
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

BPEROOT=/home/ba63/gec/subword-nmt/subword_nmt
DATA_DIR=/scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014
FAIRSEQ_DATA_DIR=/scratch/ba63/gec/fairseq-data/QALB-2014_new
# DATA_DIR=/scratch/ba63/arabic-gec-new/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014
# DATA_DIR=/scratch/ba63/arabic-gec-new/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014
BPE_TOKENS=50000
BPE_TOKENS_STR=50000

src=sent.no_ids.clean
tgt=cor.no_ids
TRAIN=$FAIRSEQ_DATA_DIR/train.cor-sent

for l in $src $tgt; do
    cat $DATA_DIR/train/QALB-2014-L1-Train.$l > $TRAIN
done


if [ -e $FAIRSEQ_DATA_DIR/bpe-${BPE_TOKENS_STR} ]; then
    printf "$FAIRSEQ_DATA_DIR/bpe-${BPE_TOKENS_STR} already exists!\n"

else
    # Learning BPE vocab over training data
    mkdir -p $FAIRSEQ_DATA_DIR/bpe-${BPE_TOKENS_STR}
    BPE_CODE=$FAIRSEQ_DATA_DIR/bpe-${BPE_TOKENS_STR}/code
    echo "learn_bpe.py on ${TRAIN}..."
    python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

    # Applying BPE tok to train, dev, tune, and test
    for L in $src $tgt; do
        for f in train/QALB-2014-L1-Train.$L dev/QALB-2014-L1-Dev.$L tune/QALB-2014-L1-Tune.$L test/QALB-2014-L1-Test.$L; do
            echo "apply_bpe.py to $DATA_DIR/$f..."
            output="$(cut -d '/' -f1 <<< ${f}).${L}"
            python $BPEROOT/apply_bpe.py -c $BPE_CODE < $DATA_DIR/$f > $FAIRSEQ_DATA_DIR/bpe-${BPE_TOKENS_STR}/$output
        done
    done

    # Binarizing the data
    fairseq-preprocess --source-lang $src --target-lang $tgt \
    --trainpref $FAIRSEQ_DATA_DIR/bpe-${BPE_TOKENS_STR}/train \
    --validpref $FAIRSEQ_DATA_DIR/bpe-${BPE_TOKENS_STR}/dev,$FAIRSEQ_DATA_DIR/bpe-${BPE_TOKENS_STR}/tune \   #This will create two valid files. valid for dev and valid1 for tune
    --testpref $FAIRSEQ_DATA_DIR/bpe-${BPE_TOKENS_STR}/test \
    --destdir $FAIRSEQ_DATA_DIR/data-bin/${BPE_TOKENS_STR}_joined \
    --workers 20 \
    --joined-dictionary

fi
