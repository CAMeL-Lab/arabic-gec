# Data

We used the [QALB-2014](), [QALB-2015](), and [ZAEBUC]() datasets to train and evaluate our models. For the QALB-2014 and QALB-2015, we use the publicly available train, dev, and test splits. For ZAEBUC, we randomly split the data into train (70%), dev (15%), and test (15%) while keeping a balanced distribution of CEFR levels. 

## Preprocessing, Alignment, and Automatic Error Typing:

### Preprocessing:

We preprocess the data before using it to train our models. The preprocessing includes removing diacritics for all erroneous and corrected sentence pairs across all datasets. For the erroneous sentences in QALB-2014, we do an additional ad-hoc preprocessing step to fix a few character-tokenized sentences by stitching all the characters together. Running `bash preprocess_gec_data.sh` applies all the preprocessing steps we did. 

### Morphological Preprocessing:

We do an additional morphological preprocessing step over the erroneous sentences only. This is done by running the contextual morphological analysis and disambiguation introduced by [Inoue et al., 2022]() and that is publicly available in [CAMeL Tools](). This step is done by running `bash camelira_gec.sh`. The generated output of this preprocessing constitutes the **Morph** system we report on in our paper in Table 4.


### Alignment:

To generate the data to train and evaluate our GED models, we obtain alignments for all erroneous and corrected sentence pairs across all datasets. Moreover, we use our alignment to create m2edits for ZAEBUC and the no-punctuation versions of QALB-2014 and QALB-2015. The m2edits are needed for GEC evaluation. However, the m2edits were **not** created manually in the case of ZAEBUC and no-punctuation versions of QALB-2014 and QALB-2015.

We describe our alignment algorithm in the paper and provide a general standalone [script](https://github.com/balhafni/arabic-gec/tree/master/alignment) that allows the use of our alignment beyond this paper. Running `bash create_alignment.sh modeling` generates the alignments we use for modeling, whereas running `bash create_alignment.sh m2` generates the alignments needed to create the m2edits. The difference between these two modes is that for modeling, we use the preprocessed data to create the alignments (i.e., no diacritics and morphological preprocessing). However, for the m2edits creation, we use the data as it is without any preprocessing to be as close as possible to the manually created m2edits. 


### Automatic Error Typing:

Once the alignments have been generated, we pass them to [ARETA](https://github.com/balhafni/arabic-gec/tree/master/areta) to obtain specific error types. The error type annotations for all datasets can be obtained by running `bash create_areta_tags.sh`.


## Grammatical Error Detection:



## Grammatical Error Correction:




1) preprocessing
2) create alignment
3) create areta tags
4) create ged data
5) create json data


## M2 Scorer Edits:
