ZAEBUC Data Splits:

1) Obtaining the meta data for all documents for both Arabic and English (arabic_docs.txt and english_docs.txt):
    * cat EN-all.extracted.corrected.analyzed.corrected-FINAL.tsv | grep "^<doc " > english_docs.txt
    * cat AR-all.extracted.corrected.analyzed.corrected-FINAL.tsv | grep "^<doc " > arabic_docs.txt

2) Creating balanced splits based on CEFR levels:
    
    * run create_splits.ipynb: Generates arabic_data_extracted.txt, english_data_extracted.txt, ar_splits.txt, and en_splits.txt.

   * ar_splits.txt and en_splits.txt contain the writer ID, the split, and the CEFR level. The arabic_data_extracted.txt and english_data_extracted.txt contain the extracted information from the meta data of the docs.

   * run data_processing.ipynb: Creates both the AR-all.alignment-FINAL.splits.tsv and the EN-all.alignment-FINAL.splits.tsv files. These files contain the aligned raw and corrected tokens for each document along with its split. It will also create the train/dev/test files at the token level for both arabic and english.

  * run create_sents.ipynb: Reads the token level files and generates the *ids, *raw, *cor, and *cefr data at the document level based on the token level files.

  * run punc_sep_data.py: Separate the punctuations of the arabic document level data (*raw and *cor) files. 
