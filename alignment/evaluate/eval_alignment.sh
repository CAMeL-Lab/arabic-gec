
printf "QALB-2014 Dev Alignment Evaluation:\n"
python evaluate.py \
    --src /home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/dev/QALB-2014-L1-Dev.sent.no_ids \
    --tgt /home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/dev/QALB-2014-L1-Dev.cor.no_ids \
    --gold_alignment /home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/dev/QALB-2014-L1-Dev.m2 \
    --m2_alignment qalb14_dev.m2 \
    --areta_alignment qalb14_dev.align \
    --our_alignment /home/ba63/gec-release/data/alignments/m2/qalb14/qalb14_dev.txt \


printf "\nQALB-2015 Dev Alignment Evaluation:\n"
python evaluate.py \
    --src /home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/dev/QALB-2015-L2-Dev.sent.no_ids \
    --tgt /home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/dev/QALB-2015-L2-Dev.cor.no_ids \
    --gold_alignment /home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/dev/QALB-2015-L2-Dev.m2 \
    --m2_alignment qalb15_dev.m2 \
    --areta_alignment qalb15_dev.align \
    --our_alignment /home/ba63/gec-release/data/alignments/m2/qalb15/qalb15_dev.txt \
