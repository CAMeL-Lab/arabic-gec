#!/bin/bash
#SBATCH -p nvidia
#SBATCH -q nlp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120GB
#SBATCH --time=47:59:00
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err

python aligner.py --src /scratch/ba63/gec/data/gec_camelira/qalb14/qalb14_train.src.txt.nopnx  \
                   --tgt /scratch/ba63/gec/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/train/QALB-2014-L1-Train.cor.no_ids.dediac.nopnx  \
                    --output /scratch/ba63/gec/data/gec_camelira/alignment/qalb14_train.txt.nopnx 

python aligner.py --src /scratch/ba63/gec/data/gec_camelira/qalb14/qalb14_tune.src.txt.nopnx  \
                    --tgt /scratch/ba63/gec/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/tune/QALB-2014-L1-Tune.cor.no_ids.dediac.nopnx  \
                    --output /scratch/ba63/gec/data/gec_camelira/alignment/qalb14_tune.txt.nopnx 

python aligner.py --src /scratch/ba63/gec/data/gec_camelira/qalb14/qalb14_dev.src.txt.nopnx  \
                    --tgt /scratch/ba63/gec/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/dev/QALB-2014-L1-Dev.cor.no_ids.dediac.nopnx  \
                    --output /scratch/ba63/gec/data/gec_camelira/alignment/qalb14_dev.txt.nopnx 

python aligner.py --src /scratch/ba63/gec/data/gec_camelira/qalb14/qalb14_test.src.txt.nopnx  \
                    --tgt /scratch/ba63/gec/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/test/QALB-2014-L1-Test.cor.no_ids.dediac.nopnx  \
                    --output /scratch/ba63/gec/data/gec_camelira/alignment/qalb14_test.txt.nopnx 

python aligner.py --src /scratch/ba63/gec/data/gec_camelira/qalb15/qalb15_train.src.txt.nopnx  \
                    --tgt /scratch/ba63/gec/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/train/QALB-2015-L2-Train.cor.no_ids.dediac.nopnx  \
                    --output /scratch/ba63/gec/data/gec_camelira/alignment/qalb15_train.txt.nopnx 

python aligner.py --src /scratch/ba63/gec/data/gec_camelira/qalb15/qalb15_dev.src.txt.nopnx  \
                    --tgt /scratch/ba63/gec/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/dev/QALB-2015-L2-Dev.cor.no_ids.dediac.nopnx  \
                    --output /scratch/ba63/gec/data/gec_camelira/alignment/qalb15_dev.txt.nopnx

python aligner.py --src /scratch/ba63/gec/data/gec_camelira/qalb15/qalb15_L2-test.src.txt.nopnx  \
                    --tgt /scratch/ba63/gec/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/test/QALB-2015-L2-Test.cor.no_ids.dediac.nopnx  \
                    --output /scratch/ba63/gec/data/gec_camelira/alignment/qalb15_L2-test.txt.nopnx 

python aligner.py --src /scratch/ba63/gec/data/gec_camelira/qalb15/qalb15_L1-test.src.txt.nopnx  \
                    --tgt /scratch/ba63/gec/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/test/QALB-2015-L1-Test.cor.no_ids.dediac.nopnx  \
                    --output /scratch/ba63/gec/data/gec_camelira/alignment/qalb15_L1-test.txt.nopnx 

python aligner.py --src /scratch/ba63/gec/data/gec_camelira/zaebuc/zaebuc_train.src.txt.nopnx  \
                    --tgt /scratch/ba63/gec/data/gec/ZAEBUC-v1.0/data/ar/train/train.sent.cor.pnx.tok.dediac.nopnx  \
                    --output /scratch/ba63/gec/data/gec_camelira/alignment/zaebuc_train.txt.nopnx 

python aligner.py --src /scratch/ba63/gec/data/gec_camelira/zaebuc/zaebuc_dev.src.txt.nopnx  \
                    --tgt /scratch/ba63/gec/data/gec/ZAEBUC-v1.0/data/ar/dev/dev.sent.cor.pnx.tok.dediac.nopnx  \
                    --output /scratch/ba63/gec/data/gec_camelira/alignment/zaebuc_dev.txt.nopnx 

python aligner.py --src /scratch/ba63/gec/data/gec_camelira/zaebuc/zaebuc_test.src.txt.nopnx  \
                    --tgt /scratch/ba63/gec/data/gec/ZAEBUC-v1.0/data/ar/test/test.sent.cor.pnx.tok.dediac.nopnx  \
                    --output /scratch/ba63/gec/data/gec_camelira/alignment/zaebuc_test.txt.nopnx 
