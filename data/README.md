# Data

We used the [QALB-2014](https://camel.abudhabi.nyu.edu/qalb-shared-task-2015/), [QALB-2015](https://camel.abudhabi.nyu.edu/qalb-shared-task-2015/), and [ZAEBUC](https://sites.google.com/view/zaebuc/home) datasets to train and evaluate our models. For the QALB-2014 and QALB-2015, we use the publicly available train, dev, and test splits. For ZAEBUC, we randomly split the data into train (70%), dev (15%), and test (15%) while keeping a balanced distribution of CEFR levels. 

## Preprocessing, Alignment, and Automatic Error Typing:

### Preprocessing:

We preprocess the data before using it to train our models. The preprocessing includes removing diacritics for all erroneous and corrected sentence pairs across all datasets. For the erroneous sentences in QALB-2014, we do an additional ad-hoc preprocessing step to fix a few character-tokenized sentences by stitching all the characters together. Running `bash preprocess_gec_data.sh` from the `utils/` directory applies all the preprocessing steps we did. 

### Morphological Preprocessing:

We do an additional morphological preprocessing step over the erroneous sentences only. This is done by running the contextual morphological analysis and disambiguation model introduced by [Inoue et al., 2022](https://aclanthology.org/2022.findings-acl.135.pdf) and that is publicly available in [CAMeL Tools](https://github.com/CAMeL-Lab/camel_tools). This step is done by running `bash camelira_gec.sh` from the `utils/` directory. The generated output of this preprocessing constitutes the **Morph** system we report on in our paper in Table 4. **Important Note**: please note that you have to use an extended version of the [SAMA 3.1](https://catalog.ldc.upenn.edu/LDC2010L01) morphological database to replicate our experiment. We [release](https://github.com/CAMeL-Lab/arabic-gec/releases/tag/arabic-gec) a [muddled](https://github.com/CAMeL-Lab/muddler) version of this database. Once you download the database, make sure to update its [path](https://github.com/CAMeL-Lab/arabic-gec/blob/master/data/utils/camelira_gec.py#L10) in the script we provide.


### Alignment:

To generate the data to train and evaluate our GED models, we obtain alignments for all erroneous and corrected sentence pairs across all datasets. Moreover, we use our alignment to create m2edits for ZAEBUC and the no-punctuation versions of QALB-2014 and QALB-2015. The m2edits are needed for GEC evaluation. However, the m2edits were **not** created manually in the case of ZAEBUC and no-punctuation versions of QALB-2014 and QALB-2015.

We describe our alignment algorithm in the paper and provide a general standalone [script](https://github.com/balhafni/arabic-gec/tree/master/alignment) that allows the use of our alignment beyond this paper. Running `bash create_alignment.sh modeling` generates the alignments we use for modeling, whereas running `bash create_alignment.sh m2` generates the alignments needed to create the m2edits. The difference between these two modes is that for modeling, we use the preprocessed data to create the alignments (i.e., no diacritics and morphological preprocessing). However, for the m2edits creation, we use the data as it is without any preprocessing to be as close as possible to the manually created m2edits. 


### Automatic Error Typing:

Once the alignments have been generated, we pass them to [ARETA](https://github.com/balhafni/arabic-gec/tree/master/areta) to obtain specific error types. The error type annotations for all datasets can be obtained by running `bash create_areta_tags.sh`.


## Grammatical Error Detection:

To obtain the grammatical error detection data, we project multi-token error type annotations to single-token labels. Running `bash create_ged_data.sh` will generate the GED data we used to train and evaluate our GED models. Regarding the GED granularity experiments, the 13-Class and 2-Class GED data can be obtained by running `bash create_ged_granularity.sh`. The GED data can be found [here](https://github.com/balhafni/arabic-gec/tree/master/data/ged). Each data directory has the following structure:

```
dataset
├── w_camelira
│ ├── binary
│ ├── coarse
│ └── full
└── wo_camelira
    ├── binary
    ├── coarse
    └── full
```

Where dataset can be qalb14, qalb15, zaebuc, qalb14-15, and mix. w_camelira and wo_camelira indicate if the GED data was obtained before (wo_camelira) or after (w_camelira) morphological preprocessing. full corresponds to 43-Class GED, coarse corresponds to 13-Class GED, and binary corresponds to 2-Class GED.

qalb14, qalb15, and zaebuc contain the train, dev, and test sets for each dataset. qalb14-15 contain the merged training set of qalb14 and qalb15, mix contain the merged training set of qalb14, qalb15, and zaebuc.




## Grammatical Error Correction:

For GEC, we create json lines files that encapsulate both the grammatical error correction and detection data. This is what we use to fine-tune and evaluate the seq2seq models we experiment with. This data can be obtained by running `bash jsonify_data.sh` and can be found [here](https://github.com/balhafni/arabic-gec/tree/master/data/gec/modeling). The data directories follow the same structure and naming conventions as the ones used for the GED data above. Each json line consists of three keys: `raw`, `cor`, and `ged_tags`, where raw is the erroneous data, cor is the corrected data, and ged_tags is the ged tags for each word in raw.

It is important to note that the GED data used to create the json lines are based on the predictions of our GED models. Except for `*.oracle.json` files where we used gold GED data, which are used to conduct our oracle experiments.


## M<sup>2</sup> Scorer Edits:

Since the ZAEBUC dataset does not have manually created m2edits files, as was done in QALB-2014 and QALB-2015, we use our alignment algorithm to generate the m2edits for the dev and test sets for ZAEBUC. Moreover, we also rely on our alignment algorithm to generate m2edits for the non-punctuation versions of the dev and test sets of QALB-2014, QALB-2015, and ZAEBUC. Automatically generated m2edits are [here](https://github.com/balhafni/arabic-gec/tree/master/data/m2edits) and can obtained by running `bash create_m2edits.sh`.
