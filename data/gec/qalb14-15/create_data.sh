qalb14_dir=/home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/train
qalb15_dir=/home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/train

cat ${qalb14_dir}/QALB-2014-L1-Train.sent.no_ids.clean.dediac  ${qalb15_dir}/QALB-2015-L2-Train.sent.no_ids.dediac > train.sent.dediac
cat ${qalb14_dir}/QALB-2014-L1-Train.cor.no_ids.dediac  ${qalb15_dir}/QALB-2015-L2-Train.cor.no_ids.dediac >  train.cor.dediac
