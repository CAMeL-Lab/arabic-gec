qalb14_dir=/home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/dev
qalb15_dir=/home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/dev


python ced_word_alignment/align_text.py \
    -s $qalb14_dir/QALB-2014-L1-Dev.sent.no_ids \
    -t $qalb14_dir/QALB-2014-L1-Dev.cor.no_ids \
    -m basic \
    -o qalb14.ced

python /home/ba63/gec-release/areta/utilities/adjust_align_tool.py qalb14.ced.basic > qalb14_dev.align.areta
sed -i '1s/^/SOURCE  TARGET\n/' qalb14_dev.align.areta
rm qalb14.ced.basic

python ced_word_alignment/align_text.py \
    -s $qalb15_dir/QALB-2015-L2-Dev.sent.no_ids \
    -t $qalb15_dir/QALB-2015-L2-Dev.cor.no_ids \
    -m basic \
    -o qalb15.ced

python /home/ba63/gec-release/areta/utilities/adjust_align_tool.py qalb15.ced.basic > qalb15_dev.align.areta
sed -i '1s/^/SOURCE  TARGET\n/' qalb15_dev.align.areta
rm qalb15.ced.basic
