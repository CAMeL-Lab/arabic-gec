# We are creating m2 files *without* dediac because we want it to be as close
# as to the m2 files that were created by the annotators (modulo punctuation).
# If we dediac, we will ignore the systems corrections on fixing dediac cases!

qalb14_dir=/home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014
qalb15_dir=/home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015
zaebuc_dir=/home/ba63/gec-release/data/gec/ZAEBUC-v1.0/data/ar

for split in dev test
do
      align_dir=/home/ba63/gec-release/data/alignments/m2/qalb14
      s="$(tr '[:lower:]' '[:upper:]' <<< ${split:0:1})${split:1}"

      python /home/ba63/gec-release/alignment/create_m2_file.py \
            --src ${qalb14_dir}/${split}/QALB-2014-L1-${s}.sent.no_ids.nopnx \
            --tgt ${qalb14_dir}/${split}/QALB-2014-L1-${s}.cor.no_ids.nopnx \
            --align ${align_dir}/qalb14_${split}.nopnx.txt \
            --output qalb14/qalb14_${split}.nopnx.m2
done


for split in dev test-L1 test-L2
do
      align_dir=/home/ba63/gec-release/data/alignments/m2/qalb15

      if [ "$split" = "test-L1" ]; then
            fname=QALB-2015-L1
            split="$(cut -d '-' -f1 <<< ${split})"
            lang=L1-

      elif [ "$split" = "test-L2" ]; then
            fname=QALB-2015-L2
            split="$(cut -d '-' -f1 <<< ${split})"
            lang=L2-
      else
            fname=QALB-2015-L2
            lang=""
      fi

      s="$(tr '[:lower:]' '[:upper:]' <<< ${split:0:1})${split:1}"

      python /home/ba63/gec-release/alignment/create_m2_file.py \
            --src ${qalb15_dir}/${split}/${fname}-${s}.sent.no_ids.nopnx \
            --tgt ${qalb15_dir}/${split}/${fname}-${s}.cor.no_ids.nopnx \
            --align ${align_dir}/qalb15_${lang}${split}.nopnx.txt \
            --output qalb15/qalb15_${lang}${split}.nopnx.m2
done



for split in dev test
do
      align_dir=/home/ba63/gec-release/data/alignments/m2/zaebuc
      s="$(tr '[:lower:]' '[:upper:]' <<< ${split:0:1})${split:1}"

      python /home/ba63/gec-release/alignment/create_m2_file.py \
            --src ${zaebuc_dir}/${split}/${split}.sent.raw.pnx.tok \
            --tgt ${zaebuc_dir}/${split}/${split}.sent.cor.pnx.tok \
            --align ${align_dir}/zaebuc_${split}.txt \
            --output zaebuc/zaebuc_${split}.m2

      python /home/ba63/gec-release/alignment/create_m2_file.py \
            --src ${zaebuc_dir}/${split}/${split}.sent.raw.pnx.tok.nopnx \
            --tgt ${zaebuc_dir}/${split}/${split}.sent.cor.pnx.tok.nopnx \
            --align ${align_dir}/zaebuc_${split}.nopnx.txt \
            --output zaebuc/zaebuc_${split}.nopnx.m2
done