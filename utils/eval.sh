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
conda activate python2

export m2_scorer=/scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/m2Scripts/m2scorer.py
# export system_hyp=/scratch/ba63/gec/models/vanilla-transformers/500_bpe/outputs
# export m2_edits=/scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/dev/QALB-2015-L2-Dev.m2
export m2_edits=/scratch/ba63/gec/data/ZAEBUC-v1.0/data/ar/dev/m2.dev.edits
export system_hyp=/scratch/ba63/gec/models/ZAEBUC/bart


if [ "$1" = "nopnx" ] && [ "$2" = "verbose" ]; then
    echo "Running M2 evaluation in verbose mode with nopnx"
    python $m2_scorer --verbose $system_hyp/hyp.txt.nopnx $m2_edits.nopnx > $system_hyp/m2.verbose.eval.nopnx

elif [ "$1" = "nopnx" ]; then
    echo "Running M2 evaluation with nopnx"
    python $m2_scorer $system_hyp/hyp.txt.nopnx $m2_edits.nopnx > $system_hyp/m2.eval.nopnx

elif [ "$1" = "verbose" ]; then
    echo "Running M2 evaluation in verbose mode"
    # python $m2_scorer --verbose $system_hyp/hyp.txt.beam_5 $m2_edits > $system_hyp/m2.verbose.eval.beam_5
    python $m2_scorer --verbose $system_hyp/generated_predictions.txt.clean.tune $m2_edits > $system_hyp/m2.eval.verbose.tune

# else
#     echo "Running M2 evaluation"
#     python $m2_scorer $system_hyp/hyp.txt $m2_edits > $system_hyp/m2.eval
#     python $m2_scorer --verbose --beta 1.0 $system_hyp/hyp.txt $m2_edits > $system_hyp/m2.verbose.eval

else
    echo "Running M2 evaluation"
    python $m2_scorer $system_hyp/dev.pred.txt.pnx.edits $m2_edits > $system_hyp/m2.dev.eval.pnx.edits
    # python $m2_scorer $system_hyp/hyp1.txt.pnx.edits $m2_edits > $system_hyp/m2.tune.eval.pnx.edits
    # python $m2_scorer $system_hyp/tune.pred.txt.pnx.edits $m2_edits > $system_hyp/m2.tune.eval.pnx.edits
    

fi
