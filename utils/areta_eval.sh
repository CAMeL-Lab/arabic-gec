system_output=/scratch/ba63/gec/models/MIX/t5_new/t5_with_full_areta_30/
# system_output=/scratch/ba63/gec/models/MIX/t5_new/t5_with_full_areta_30/checkpoint-65000
# m2_file=/scratch/ba63/gec/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/dev/QALB-2015-L2-Dev.m2
m2_file=/scratch/ba63/gec/data/ZAEBUC-v1.0/data/ar/dev/m2.dev.edits

pred_file=dev.zaebuc.pred.txt.pnx.edits

cat $system_output/$pred_file | sed 's/^/s /g' > $system_output/$pred_file.areta

python /home/ba63/arabic_error_type_annotation/annotate_eval_ar.py \
    $system_output/$pred_file.areta \
    $m2_file \
    $system_output/areta_eval

rm $system_output/$pred_file.areta



