# Data

We used the [QALB-2014](), [QALB-2015](), and [ZAEBUC]() datasets to train and evaluate our models. For the QALB-2014 and QALB-2015, we use the publicly available train, dev, and test splits. For ZAEBUC, we randomly split the data into train (70%), dev (15%), and test (15%) while keeping a balanced distribution of CEFR levels. 

## Preprocessing, Alignment, and Automatic Error Typing:

### Preprocessing:

We preprocess the data before using it to train our models. The preprocessing includes removing diacritics for all erroneous and corrected sentence pairs across all datasets. For the erroneous sentences in QALB-2014, we do an additional ad-hoc preprocessing step to fix a few sentences that are character-tokenized by stitching all the characters together. Running `bash preprocess_gec_data.sh` applies all the preprocessing steps we did. 

### Morphological Preprocessing:

We do an additional morphological preprocessing step over the erroneous sentences only. This is done by running the contextual morphological analysis and disambiguation introduced by [Inoue et al., 2022]() and that is publicly available in [CAMeL Tools](). This step is done by running `bash camelira_gec.sh`. The generated output of this preprocessing constitutes the **Morph** system we report on in our paper in Table 4.

### Alignment:

## Grammatical Error Correction:




## Grammatical Error Detection:

1) preprocessing
2) create alignment
3) create areta tags
4) create ged data
5) create json data


## M2 Scorer Edits:
