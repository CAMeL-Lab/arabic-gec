python create_m2_file.py \
      --src /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/tune/QALB-2014-L1-Tune.sent.no_ids.clean \
      --tgt /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/tune/QALB-2014-L1-Tune.cor.no_ids \
      --align /scratch/ba63/gec/data/alignment/m2_files_alignment/qalb14/qalb14_tune.txt \
      --output /scratch/ba63/gec/data/alignment/m2_files/qalb14_tune.m2

python create_m2_file.py \
      --src /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/dev/QALB-2014-L1-Dev.sent.no_ids \
      --tgt /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/dev/QALB-2014-L1-Dev.cor.no_ids \
      --align /scratch/ba63/gec/data/alignment/m2_files_alignment/qalb14/qalb14_dev.txt \
      --output /scratch/ba63/gec/data/alignment/m2_files/qalb14_dev.m2

python create_m2_file.py \
      --src /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/test/QALB-2014-L1-Test.sent.no_ids \
      --tgt /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/test/QALB-2014-L1-Test.cor.no_ids \
      --align /scratch/ba63/gec/data/alignment/m2_files_alignment/qalb14/qalb14_test.txt \
      --output /scratch/ba63/gec/data/alignment/m2_files/qalb14_test.m2


python create_m2_file.py \
      --src /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/dev/QALB-2015-L2-Dev.sent.no_ids \
      --tgt /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/dev/QALB-2015-L2-Dev.cor.no_ids \
      --align /scratch/ba63/gec/data/alignment/m2_files_alignment/qalb15/qalb15_dev.txt \
      --output /scratch/ba63/gec/data/alignment/m2_files/qalb15_dev.m2

python create_m2_file.py \
      --src /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/test/QALB-2015-L2-Test.sent.no_ids \
      --tgt /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/test/QALB-2015-L2-Test.cor.no_ids \
      --align /scratch/ba63/gec/data/alignment/m2_files_alignment/qalb15/qalb15_L2-test.txt \
      --output /scratch/ba63/gec/data/alignment/m2_files/qalb15_L2-test.m2

python create_m2_file.py \
      --src /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/test/QALB-2015-L1-Test.sent.no_ids \
      --tgt /scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/test/QALB-2015-L1-Test.cor.no_ids \
      --align /scratch/ba63/gec/data/alignment/m2_files_alignment/qalb15/qalb15_L1-test.txt \
      --output /scratch/ba63/gec/data/alignment/m2_files/qalb15_L1-test.m2

python create_m2_file.py \
      --src /scratch/ba63/gec/data/ZAEBUC-v1.0/data/ar/dev/dev.sent.raw.pnx.tok \
      --tgt /scratch/ba63/gec/data/ZAEBUC-v1.0/data/ar/dev/dev.sent.cor.pnx.tok \
      --align /scratch/ba63/gec/data/alignment/m2_files_alignment/zaebuc/zaebuc_dev.txt \
      --output /scratch/ba63/gec/data/alignment/m2_files/zaebuc_dev.m2

python create_m2_file.py \
      --src /scratch/ba63/gec/data/ZAEBUC-v1.0/data/ar/test/test.sent.raw.pnx.tok \
      --tgt /scratch/ba63/gec/data/ZAEBUC-v1.0/data/ar/test/test.sent.cor.pnx.tok \
      --align /scratch/ba63/gec/data/alignment/m2_files_alignment/zaebuc/zaebuc_test.txt \
      --output /scratch/ba63/gec/data/alignment/m2_files/zaebuc_test.m2
