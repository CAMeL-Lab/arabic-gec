python create_m2_file.py \
      --src /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/tune/QALB-2014-L1-Tune.sent.no_ids.clean.nopnx \
      --tgt /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/tune/QALB-2014-L1-Tune.cor.no_ids.nopnx \
      --align /scratch/ba63/gec/data/alignment/m2_files_alignment/qalb14/qalb14_tune.nopnx.txt \
      --output /scratch/ba63/gec/data/alignment/m2_files/qalb14_tune.nopnx.m2

python create_m2_file.py \
      --src /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/dev/QALB-2014-L1-Dev.sent.no_ids.nopnx \
      --tgt /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/dev/QALB-2014-L1-Dev.cor.no_ids.nopnx \
      --align /scratch/ba63/gec/data/alignment/m2_files_alignment/qalb14/qalb14_dev.nopnx.txt \
      --output /scratch/ba63/gec/data/alignment/m2_files/qalb14_dev.nopnx.m2

python create_m2_file.py \
      --src /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/test/QALB-2014-L1-Test.sent.no_ids.nopnx \
      --tgt /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/test/QALB-2014-L1-Test.cor.no_ids.nopnx \
      --align /scratch/ba63/gec/data/alignment/m2_files_alignment/qalb14/qalb14_test.nopnx.txt \
      --output /scratch/ba63/gec/data/alignment/m2_files/qalb14_test.nopnx.m2


python create_m2_file.py \
      --src /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/dev/QALB-2015-L2-Dev.sent.no_ids.nopnx \
      --tgt /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/dev/QALB-2015-L2-Dev.cor.no_ids.nopnx \
      --align /scratch/ba63/gec/data/alignment/m2_files_alignment/qalb15/qalb15_dev.nopnx.txt \
      --output /scratch/ba63/gec/data/alignment/m2_files/qalb15_dev.nopnx.m2

python create_m2_file.py \
      --src /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/test/QALB-2015-L2-Test.sent.no_ids.nopnx \
      --tgt /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/test/QALB-2015-L2-Test.cor.no_ids.nopnx \
      --align /scratch/ba63/gec/data/alignment/m2_files_alignment/qalb15/qalb15_L2-test.nopnx.txt \
      --output /scratch/ba63/gec/data/alignment/m2_files/qalb15_L2-test.nopnx.m2

python create_m2_file.py \
      --src /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/test/QALB-2015-L1-Test.sent.no_ids.nopnx \
      --tgt /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/test/QALB-2015-L1-Test.cor.no_ids.nopnx \
      --align /scratch/ba63/gec/data/alignment/m2_files_alignment/qalb15/qalb15_L1-test.nopnx.txt \
      --output /scratch/ba63/gec/data/alignment/m2_files/qalb15_L1-test.nopnx.m2

python create_m2_file.py \
      --src /scratch/ba63/gec/data/ZAEBUC-v1.0/data/ar/dev/dev.sent.raw.pnx.tok.nopnx \
      --tgt /scratch/ba63/gec/data/ZAEBUC-v1.0/data/ar/dev/dev.sent.cor.pnx.tok.nopnx \
      --align /scratch/ba63/gec/data/alignment/m2_files_alignment/zaebuc/zaebuc_dev.nopnx.txt \
      --output /scratch/ba63/gec/data/alignment/m2_files/zaebuc_dev.nopnx.m2

python create_m2_file.py \
      --src /scratch/ba63/gec/data/ZAEBUC-v1.0/data/ar/test/test.sent.raw.pnx.tok.nopnx \
      --tgt /scratch/ba63/gec/data/ZAEBUC-v1.0/data/ar/test/test.sent.cor.pnx.tok.nopnx \
      --align /scratch/ba63/gec/data/alignment/m2_files_alignment/zaebuc/zaebuc_test.nopnx.txt \
      --output /scratch/ba63/gec/data/alignment/m2_files/zaebuc_test.nopnx.m2
