## Maximum Likelihood Estimation (MLE)


The MLE model we describe in our paper exploits our alignment algorithm and the error type annotations obtained using ARETA to map erroneous words to their corrections.
Running the MLE model is done using the [rewriter.sh](rewriter.sh) script:

```bash
train_file=/path/to/train/file
test_file=/path/to/test/file
ged_model=/path/to/ged/model
output_path=/path/to/output/file

python rewriter.py \
  --train_file $train_file \
  --test_file  $test_file \
  --ged_model  $ged_model \
  --mode full \
  --cbr_ngrams 2 \
  --output_path $output_path
```
