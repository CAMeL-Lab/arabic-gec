for model in ./*
do

      # mv $model/qalb14_dev.preds.check.txt.pp $model/qalb14_dev.preds.txt.pp 
      #mv $model/qalb14_dev.preds.check.txt $model/qalb14_dev.preds.txt


      # mv $model/qalb14_dev.preds.check.txt.nopnx.pp $model/qalb14_dev.preds.txt.nopnx.pp
      # mv $model/qalb14_dev.preds.check.txt.nopnx $model/qalb14_dev.preds.txt.nopnx


      # mv $model/qalb14_dev.preds.check.txt.m2 $model/qalb14_dev.preds.txt.m2
      # mv $model/qalb14_dev.preds.check.txt.nopnx.m2   $model/qalb14_dev.preds.txt.nopnx.m2


      # mv $model/qalb14_test.preds.check.txt $model/qalb14_test.preds.txt

      # mv $model/qalb14_test.preds.check.txt.nopnx $model/qalb14_test.preds.txt.nopnx

      # mv $model/qalb14_test.preds.txt.official.m2 $model/qalb14_test.preds.txt.m2
      # mv $model/qalb14_test.preds.txt.nopnx.official.m2  $model/qalb14_test.preds.txt.nopnx.m2 
      
      if [ -f "${model}/qalb14_dev.preds.txt.pp" ]; then
           mv ${model}/qalb14_dev.preds.txt.m2 ${model}/qalb14_dev.preds.txt.pp.m2
      fi

      if [ -f "${model}/qalb14_dev.preds.oracle.txt.pp" ]; then
           mv ${model}/qalb14_dev.preds.oracle.txt.m2 ${model}/qalb14_dev.preds.oracle.txt.pp.m2
      fi
done
