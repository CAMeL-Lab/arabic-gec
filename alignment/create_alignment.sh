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

if [ "$1" = "m2" ]; then
    python aligner.py --src /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/tune/QALB-2014-L1-Tune.sent.no_ids.clean \
                      --tgt /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/tune/QALB-2014-L1-Tune.cor.no_ids \
                      --output /scratch/ba63/gec/data/alignment/m2_files_alignment/qalb14/qalb14_tune.txt

    python aligner.py --src /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/dev/QALB-2014-L1-Dev.sent.no_ids \
                      --tgt /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/dev/QALB-2014-L1-Dev.cor.no_ids \
                      --output /scratch/ba63/gec/data/alignment/m2_files_alignment/qalb14/qalb14_dev.txt

    python aligner.py --src /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/test/QALB-2014-L1-Test.sent.no_ids \
                      --tgt /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/test/QALB-2014-L1-Test.cor.no_ids \
                      --output /scratch/ba63/gec/data/alignment/m2_files_alignment/qalb14/qalb14_test.txt

    python aligner.py --src /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/dev/QALB-2015-L2-Dev.sent.no_ids \
                      --tgt /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/dev/QALB-2015-L2-Dev.cor.no_ids \
                      --output /scratch/ba63/gec/data/alignment/m2_files_alignment/qalb15/qalb15_dev.txt

    python aligner.py --src /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/test/QALB-2015-L2-Test.sent.no_ids \
                      --tgt /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/test/QALB-2015-L2-Test.cor.no_ids \
                      --output /scratch/ba63/gec/data/alignment/m2_files_alignment/qalb15/qalb15_L2-test.txt

    python aligner.py --src /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/test/QALB-2015-L1-Test.sent.no_ids \
                      --tgt /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/test/QALB-2015-L1-Test.cor.no_ids \
                      --output /scratch/ba63/gec/data/alignment/m2_files_alignment/qalb15/qalb15_L1-test.txt

    python aligner.py --src /scratch/ba63/gec/data/ZAEBUC-v1.0/data/ar/dev/dev.sent.raw.pnx.tok \
                      --tgt /scratch/ba63/gec/data/ZAEBUC-v1.0/data/ar/dev/dev.sent.cor.pnx.tok \
                      --output /scratch/ba63/gec/data/alignment/m2_files_alignment/zaebuc/zaebuc_dev.txt

    python aligner.py --src /scratch/ba63/gec/data/ZAEBUC-v1.0/data/ar/test/test.sent.raw.pnx.tok \
                      --tgt /scratch/ba63/gec/data/ZAEBUC-v1.0/data/ar/test/test.sent.cor.pnx.tok \
                      --output /scratch/ba63/gec/data/alignment/m2_files_alignment/zaebuc/zaebuc_test.txt

# CREATE THE ALIGNMENT WE NEED FOR MODELING (i.e., removing diacs and kashidas)
elif [ "$1" = "modeling" ]; then
    python aligner.py --src /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/train/QALB-2014-L1-Train.sent.no_ids.clean.dediac \
                      --tgt /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/train/QALB-2014-L1-Train.cor.no_ids.dediac \
                      --output /scratch/ba63/gec/data/alignment/modeling_alignment/qalb14/qalb14_train.txt

    python aligner.py --src /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/tune/QALB-2014-L1-Tune.sent.no_ids.clean.dediac \
                      --tgt /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/tune/QALB-2014-L1-Tune.cor.no_ids.dediac \
                     --output /scratch/ba63/gec/data/alignment/modeling_alignment/qalb14/qalb14_tune.txt

    python aligner.py --src /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/dev/QALB-2014-L1-Dev.sent.no_ids.clean.dediac \
                      --tgt /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/dev/QALB-2014-L1-Dev.cor.no_ids.dediac \
                      --output /scratch/ba63/gec/data/alignment/modeling_alignment/qalb14/qalb14_dev.txt

    python aligner.py --src /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/test/QALB-2014-L1-Test.sent.no_ids.clean.dediac \
                      --tgt /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/test/QALB-2014-L1-Test.cor.no_ids.dediac \
                      --output /scratch/ba63/gec/data/alignment/modeling_alignment/qalb14/qalb14_test.txt

    python aligner.py --src /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/train/QALB-2015-L2-Train.sent.no_ids.dediac \
                      --tgt /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/train/QALB-2015-L2-Train.cor.no_ids.dediac \
                      --output /scratch/ba63/gec/data/alignment/modeling_alignment/qalb15/qalb15_train.txt

    python aligner.py --src /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/dev/QALB-2015-L2-Dev.sent.no_ids.dediac \
                      --tgt /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/dev/QALB-2015-L2-Dev.cor.no_ids.dediac \
                      --output /scratch/ba63/gec/data/alignment/modeling_alignment/qalb15/qalb15_dev.txt

    python aligner.py --src /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/test/QALB-2015-L2-Test.sent.no_ids.dediac \
                      --tgt /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/test/QALB-2015-L2-Test.cor.no_ids.dediac \
                      --output /scratch/ba63/gec/data/alignment/modeling_alignment/qalb15/qalb15_L2-test.txt

    python aligner.py --src /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/test/QALB-2015-L1-Test.sent.no_ids.dediac \
                      --tgt /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/test/QALB-2015-L1-Test.cor.no_ids.dediac \
                      --output /scratch/ba63/gec/data/alignment/modeling_alignment/qalb15/qalb15_L1-test.txt

    python aligner.py --src /scratch/ba63/gec/data/ZAEBUC-v1.0/data/ar/train/train.sent.raw.pnx.tok.dediac \
                      --tgt /scratch/ba63/gec/data/ZAEBUC-v1.0/data/ar/train/train.sent.cor.pnx.tok.dediac \
                      --output /scratch/ba63/gec/data/alignment/modeling_alignment/zaebuc/zaebuc_train.txt
    
    python aligner.py --src /scratch/ba63/gec/data/ZAEBUC-v1.0/data/ar/dev/dev.sent.raw.pnx.tok.dediac \
                      --tgt /scratch/ba63/gec/data/ZAEBUC-v1.0/data/ar/dev/dev.sent.cor.pnx.tok.dediac \
                      --output /scratch/ba63/gec/data/alignment/modeling_alignment/zaebuc/zaebuc_dev.txt

    python aligner.py --src /scratch/ba63/gec/data/ZAEBUC-v1.0/data/ar/test/test.sent.raw.pnx.tok.dediac \
                      --tgt /scratch/ba63/gec/data/ZAEBUC-v1.0/data/ar/test/test.sent.cor.pnx.tok.dediac \
                      --output /scratch/ba63/gec/data/alignment/modeling_alignment/zaebuc/zaebuc_test.txt

fi
