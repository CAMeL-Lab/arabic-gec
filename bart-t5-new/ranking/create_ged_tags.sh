model_dir=/scratch/ba63/gec/models/gec/qalb14/bart_w_ged

for ranking_dir in $model_dir/ranking $model_dir/checkpoint-*/ranking
do
    for i in {1..5}
    do
        f=$ranking_dir/src.to.${i}.areta+.txt
        printf "$f \n"
        python /scratch/ba63/gec/data/ged/create_ged_data.py \
              --input $f \
              --output $ranking_dir/src.to.${i}.areta+.txt.ged
    done
done

