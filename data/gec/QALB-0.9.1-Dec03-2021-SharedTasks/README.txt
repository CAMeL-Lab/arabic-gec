====================================================================
QALB: Qatar Arabic Language Bank - 2014 & 2015 Shared Task Data

Current Release: 0.9.1 03 December 2021
====================================================================

Reelease Notes

-------------------------
Update 0.9.1 03 Dec 2021:
This release builds on the QALB 2015 shared task release (0.9.0) and
extends it.

   (a) Renamed files consistently (see the
       QALB-Release-Naming-Conventions.xlsx inside the docs folder).

   (b) Reorganized the data folder (created train, dev, test
       sub-folders for QALB-2014 and QALB-2015 shared tasks).

   (c) Added the corrected sentence files (e.g.,
       QALB-2014-L1-Train.cor).

   (d) Added the submitted systems' output for QALB-2014 and QALB-2015
       shared tasks (see submitted_systems folder).

   (e) Corrected the corrupted L2-QALB2015-test file from 0.9.0.

   (f) Added some helpful scripts for data processing and evaluation.

This release was created by Nizar Habash and Riadh Belkebir at New
York University Abu Dhabi - Computational Approaches to Modeling
Language (CAMeL) Lab.

-------------------------
Update 0.9.0 04 May 2021:
Added the QALB 2015 test set references for (Alj-QALB2015-test and
L2-QALB2015-test); and added the shared task description papers.

-------------------------
First Release 0.8.0 10 April 2015: 
QALB Shared Task 2015 data.

====================================================================
Copyright (c) 2015 Columbia University and Carnegie Mellon University
Qatar.  All rights reserved.
====================================================================

0. How to get this resource
=====================
To get this resource, you need to register here:
https://camel.abudhabi.nyu.edu/qalb-shared-task- 2015/.

1. Introduction
===============

This README file describes the content of the package.  The data in
this package includes portions of the QALB (Qatar Arabic Language
Bank) Corpus of Arabic intended for the shared tasks on Automatic
Arabic Error Correction: QALB 2014 and QALB 2015 Shared Tasks.

QALB was created as part of a collaborative project between Columbia
University and the Carnegie Mellon University Qatar (CMUQ) funded by
the Qatar National Research Fund (a member of the Qatar Foundation),
grant NPRP-4-1058-1-168.  This special release of QALB data for the
two Shared tasks is intended to be an archival release. We include all
of the test sets and their corrections, as well as a number of the
competing system outputs.  The QALB project page is
http://nlp.qatar.cmu.edu/qalb/.

The QALB 2014 Shared Task was conducted as part of the Workshop on
Arabic Natural Language Processing in EMNLP 2014 (Doha, Qatar). The
QALB 2014 focused on data from native speakers (commentaries written
in response to Al Jazeera articles).  Shared Task Page:
http://emnlp2014.org/workshops/anlp/shared_task.html

QALB-2015 is an extension of the QALB-2014 shared task; it was
conducted as part of the Workshop on Arabic Natural Language
Processing in with ACL-IJCNLP 2015 (Beijing, China).  The data of QALB
2015 included data from native speakers (commentaries written in
response to Al Jazeera articles) and from non-native Arabic speakers.
Shared Task
Page:https://sites.google.com/site/arabicnlpnetwork/wanlp/qalbfaq

The release is organized by the shared task year; as such, some
repetition between 2014 and 2015 is present since the Al Jazeera data
in QALB 2015 is the same data that was used for QALB 2014.

The corpus is distributed under the standard licensing agreement
available when downloading the corpus.

Any questions regarding the corpus should be directed to 
   Nizar Habash at: nizar.habash@nyu.edu.
   or Behrang Mohit at: behrangm@berkeley.edu.

If you are using the QALB corpus in your work, please cite the
following papers:

   1. Rozovskaya, Alla, Houda Bouamor, Nizar Habash, Wajdi Zaghouani,
      Ossama Obeid, and Behrang Mohit. "The second qalb shared task on
      automatic text correction for Arabic." In Proceedings of the
      Second workshop on Arabic natural language processing,
      pp. 26-35. 2015. [https://www.aclweb.org/anthology/W15-3204.pdf]

   2. Zaghouani, Wajdi, Nizar Habash, Houda Bouamor, Alla Rozovskaya,
      Behrang Mohit, Abeer Heider, and Kemal Oflazer. "Correction
      annotation for non-native Arabic texts: Guidelines and corpus."
      In Proceedings of The 9th Linguistic Annotation Workshop,
      pp. 129-139. 2015.  [https://aclanthology.org/W15-1614.pdf]

   3. Mohit, Behrang, Alla Rozovskaya, Nizar Habash, Wajdi Zaghouani,
      and Ossama Obeid. "The first QALB shared task on automatic text
      correction for Arabic." In Proceedings of the EMNLP 2014
      Workshop on Arabic Natural Language Processing (ANLP),
      pp. 39-47. 2014.  [https://aclanthology.org/W14-3605.pdf]

   4. Wajdi Zaghouani, Behrang Mohit, Nizar Habash, Ossama Obeid, Nadi
      Tomeh, Alla Rozovskaya, Noura Farra, Sarah Alkuhlani and Kemal
      Oflazer. "Large-scale Arabic Error Annotation: Guidelines and
      Framework." In Proceedings of the 9th Conference on Language
      Resources and Evaluation Conference
      (LREC-2014).
      [http://www.lrec-conf.org/proceedings/lrec2014/pdf/956_Paper.pdf]

   5. Ossama Obeid, Wajdi Zaghouani, Behrang Mohit, Nizar Habash,
      Kemal Oflazer and Nadi Tomeh, 2013. A Web-based Annotation
      Framework For Large-Scale Text Correction. In Proceedings of the
      6th International Joint Conference on Natural Language
      Processing. (IJCNLP-2013).
      [https://aclanthology.org/I13-2001.pdf]

2. File list
============

The package includes five directories:

************
*** DATA ***
************

The directory data/ has two subfolders (2014 and 2015) corresponding
to the the QALB-2014 and QALB-2015 shared tasks, respectively. Each
subfolder had three subfolders (train, dev and test) corresponding to
training, development and test data sets, respectively. Each of these,
has four files:

   (1) *.m2 files: Files in the format required by the scorer used for
       evaluation

   (2) *.sent files: All files with document IDs

   (3) *column files: Feature files in column format generated using
       the MADAMIRA morphological analysis and disambiguation of
       Arabic system (Pasha et al, 2014)

   (4) *.cor files: The reference sentences

The file names specify (a) the QALB shared task year; (b) the type of
language (L1 is native speakers' Arabic from Al Jazeera comments, and
L2 is non-native speakers' Arabic); (c) the data split (Train, Dev and
Test); and file type. For example, QALB-2015-L1-Test.m2.

*****************
*** DOCUMENTS *** 
*****************

The directory docs/ contains these files:

   (1) LREC2014.pdf: The LREC 2014 paper by Zaghouani et al (2014)
       describing the corpus creation process.

   (2) MADAMIRA-UserManual.pdf: The MADAMIRA manual.

   (3) QALB-guidelines_0.90.pdf: The annotation guidelines.

   (4) QALB-2014-Shared-Task.pdf: QALB 2014 Shared Task Description

   (5) QALB-2015-Shared-Task.pdf: QALB 2015 Shared Task Description

   (6) QALB-Release-Naming-Conventions.xlsx: The naming convention of
       the different QALB dataset releases.

**************************
*** EVALUATION SCRIPTS *** 
**************************

The directory m2Scripts/ includes scripts for running M2Scoring
evaluation (Dahlmeier and Ng, 2012).  The M2Scorer evaluation script
included with this release was used in the CoNLL-2013 shared task on
grammatical error correction (Ng et al., 2013).

Note that the script requires Python v2.7. Some additional scripts in
this directory uses Perl and some Python 3.  We have maintained the
variations for archival reasons.

    ### M2Scorer

To evaluate the performance using M2Scorer, generate a text file that
contains the corrected documents one document per line and run the
following commands, where sample.hyp is the output of your system and
sample.m2 is the sample file with gold annotations included with this
release.

Note that sample.hyp has sentence ids which need to be removed. If the
system output does not have sentence ids, then this step should be
skipped.

    python removeIDs.py sample.hyp > sample.hyp-no-id

To evaluate with M2Scorer, run:

    python m2scorer.py --verbose sample.hyp-no-id sample.m2 > sample.score

   ### QALB-Scorer

The QALB-Scorer.pl is a Perl script that generates results for Exact
Match, Normalized A/Y, No Punctuation, No Punctuation and Normalized
A/Y. To use it, run:

    perl QALB-Scorer.pl sample.hyp-no-id sample.m2 utf8

This code generates a number of intermediate files with the suffixes
(.nopnx, .norm, and.  .nopnx.norm).

   ### M2 Source and Reference

We also provide scripts to get source and reference sentences from the .m2 files:

   python generate-m2-reference.py  sample.m2 > sample.ref
   python generate-m2-source.py sample.m2 > sample.source

These files come with a sentence ids that should be removed:

   python removeIDs.py sample.ref > sample.ref-no-id
   python removeIDs.py sample.source > sample.source-no-id

*************************
*** SAMPLE TEST FILES *** 
*************************

The directory sampleTestFiles/ contains the sample files provided as
examples during the shared tasks. We have kept this directory as is
for archival reasons.

**************************
*** SYSTEM SUBMISSIONS ***
**************************

The directory submitted_systems/ contains system submissions for both
QALB-2014 and QALB-2015 shared tasks. The subdirectories specify the
tracks (L1 or L2). Each team has a subdirectory with their team name
(see the Shared Task description papers in the docs directory).


3. Data format (M2 files)
=========================

The corpus is distributed in text format.  Each file comes in three
versions: plain documents with document IDs (*sent); documents with
gold annotations (*m2), and feature files (*column).

In the *m2 files, each document is followed by the corrections that
refer to it. A correction is indicated by start position and end
position of the original string of tokens, the annotation type, and
the replacement. The format is done in the spirit of annotation style
used in the CoNLL-2013 shared task on error correction in English (Ng
et al., 2013).

Below we show an example sentence and its corresponding corrections.
In this example, the first line (S) is a document token (where a
document can be a single sentence or a paragraph of multiple sentences
written on a single line).  The following lines that start with A are
corrections referring to the document. Each correction line specifies
three important pieces of information (the fields 'REQUIRED', '-NONE-'
and '0' should be ignored):

   (a) The ID of the token in the document based on its location
       (starting at 0)
    
   (b) The type of correction (e.g. edit/merge/add_token_before)

   (c) The replacement, separated by three vertical bars (|||) on each
       side. Empty correction (i.e. deleting of target token) is
       indicated by six bars.


------------------------------------------- EXAMPLE --------------------------------------------------------
S اعداد القتلى في صفوف الارهابيين بالمئات و من الصعب حصر اعداد القتلى بشكل دقيق بسبب الحرب الشاملة التي يشنها الجيش العربي السوري في اماكن تواجدهم .
A 0 1|||edit|||أعداد|||REQUIRED|||-NONE-|||0
A 4 5|||edit|||الإرهابيين|||REQUIRED|||-NONE-|||0
A 6 6|||add_token_before|||،|||REQUIRED|||-NONE-|||0
A 6 8|||merge|||ومن|||REQUIRED|||-NONE-|||0
A 10 11|||edit|||أعداد|||REQUIRED|||-NONE-|||0
A 23 24|||edit|||أماكن|||REQUIRED|||-NONE-|||0

Target sentence:

.أعداد القتلى في صفوف الإرهابيين بالمئات، ومن الصعب حصر أعداد القتلى بشكل دقيق بسبب الحرب الشاملة التي يشنها الجيش العربي السوري في أماكن تواجدهم
--------------------------------------------------------------------------------------------------------------


* The first correction replaces token with ID 0 with the word (أعداد)
* The third correction (add_token) specifies an insertion of a comma
  in front of token with tokenID 6.
* The fourth correction merges tokens 6 and 7. The complete set of
  correction type (actions) are listed below.

A sequence of changes that replace tokens indicated in the correction
lines should produce the resulting target sentence, as shown above
(note: the target sentence is not part of the input files and is shown
here for illustration).

4. Data format (column files)
==============================

We have pre-processed the data with the MADAMIRA morphological
analyzer version 04092014-1.0-beta (see MADAMIRA-UserManual.pdf in the
docs/ directory for more detail; also see Pasha et al. (2014)). The
column files contain one word per line. Each document is followed by
an empty line. Each word has thirty-three columns of features
associated with it:

(1) Document ID which specifies the document the word appears in.
(2) Word ID which specifies the position of the word in the document.
(3) The word (e.g., وللأبد wll>bd) [example in Arabic script and in
    the Buckwalter transliteration].

The rest of the features are the output of the MADAMIRA system which
disambiguates automatically in context. MADAMIRA determines the word
lemma, diacritization, English gloss and tokenization in addition to
different types of part-of-speech tags and morphological features. The
tokenization we chose for the data is the Penn Arabic Treebank
Tokenization (PATB) tokenization which splits all word clitics except
for the definite article Al.  The tokens are joined with a + delimiter
in this file.

(4) The undiacritized word in PATB tokenization (e.g., و+ل+الأبد w+l+Al>bd)
(5) Same as (4) but Alif/Ya normalized (e.g., و+ل+الابد w+l+AlAbd)
(6) The PATB token lemmas (e.g., وَ+لِ+أَبَد wa+li+>abad)
(7) CATiB POS tag (e.g., PRT+PRT+NOM)
(8) Kulick POS tag (e.g., CC+IN+DT+NN) 
(9) Buckwalter POS tag (e.g., CONJ+PREP+DET+NOUN+CASE_DEF_GEN)
(10) MADAMIRA score for this analysis (e.g., *0.893910 )
(11) Fully diacritized word (e.g. وَلِلأَبَدِ walil>abadi)
(12) Lemma (e.g., ﺄَﺑَﺩ_1 >abad_1)
(13) Buckwalter tag with specified morphemes (e.g.,
     wa/CONJ+li/PREP+Al/DET+>abad/NOUN+i/CASE_DEF_GEN)
(14) Undiacritized form of (11) (e.g., ﻮﻟﻸﺑﺩ wll>bd) For some common
     classes of spelling errors (e.g., Alif hamzation), this is the
     correct answer. In this example, the word is already correct.
(15) Prefix gloss (e.g., and_+_to/for_+_the)
(16) Stem gloss (e.g., eternity;forever)
(17) Suffix gloss (e.g., [def.gen.])
(18) MADAMIRA core POS tag (e.g., noun)
(19) Proclitic 3 (typically interrogative article) (e.g., 0 -> not present)
(20) Proclitic 2 (e.g., wa_conj)
(21) Proclitic 1 (e.g., li_prep)
(22) Proclitic 0 (e.g., Al_det)
(23) Person (a feature of verbs; na in this example)
(24) Aspect (a feature of verbs; na in this example)
(25) Voice (a feature of verbs; na in this example)
(26) Mood (a feature of verbs; na in this example)
(27) Gender (m in this example -> masculine)
(28) Number (s in this example -> singular)
(29) State (d in this example -> definite)
     Note: the gender and number features are what Habash (2010) calls
     form-based features not functional features.
(30) Case (g in this example -> genitive)
(31) Enclitic 0 (typically a pronoun that is possessive or direct
     object; but in this example, it is not present -> 0)
(32) Lexical lookup category: This feature provides feedback from the
     analyzer on how the word was matched to the databases: lex means
     it was an exact lexicon match, spvar means it is a spelling
     variant. Other values include punc (punctuation) and digit.
(33) Stem of the word as it appears in the lexical databases used in
     MADAMIRA (e.g., أَبَد >abad)

For more information on these features and tag sets, see Habash (2010).


5. Complete set of correction types (actions)
==============================================

The annotations were carried out as a set of actions and include the
following changes:

   (1) Add_before -- insert a token in front of another token
   (2) Add_after -- insert a token after another token (rarely used)
   (3) Merge -- merge multiple tokens
   (4) Split -- split a token into multiple tokens
   (5) Delete -- delete a token
   (6) Edit - replace a token with a different token
   (7) Move - move a token to a different location in the sentence
   (8) Other - a complex action that may involve multiple tokens

The annotation guidelines can be found in the doc/ directory


6. Submission format and instructions 
=====================================

This section is maintained for archival reasons.
No submissions are accepted any more.

*** Submission Instructions ***

All those who registered to participate in the Shared Task will
receive an email message on May 16, 2015 with specific instructions on
how to download the test set and how to send the automatic correction
of it. The information will also be available at the shared task group
(https://groups.google.com/forum/#!forum/qalb-shared-task).

*** Submission Format ***

The test data will be provided to the participants in two files:
test.column and test.sent (see sample test files in sampleTestFiles/:
testSample.column and testSample.sent). These files are in the same
format as the development and training data described above.
Important: test.m2 file (the gold answers) will not be provided.

The participants will need to submit a file that contains corrected
documents one document per line. The format is the same as the file
sampleTestFiles/testSample.sent. Note that the file to be submitted
needs to specify document ID for each sentence, in the same way as the
*sent files.

Each participating team can submit up to three systems. Further
instructions on file names and where to send the submissions will be
provided to the participants.


References:
===========

(1) H. T. Ng, S. M. Wu, Y. Wu, Ch. Hadiwinoto, J. Tetreault. The
    CoNLL-2013 Shared Task on Grammatical Error Correction. In
    Proceedings of the CoNLL-2013 shared task.

(2) D. Dahlmeier and H. T. Ng. Better Evaluation for Grammatical Error
    Correction. In Proceedings of NAACL (2012).

(3) A. Pasha, M. Al-Badrashiny, A. E. Kholy, R. Eskander, M. Diab,
    N. Habash, M. Pooleery, O. Rambow, and R. Roth. MADAMIRA: A fast,
    comprehensive tool for morphological analysis and disambiguation
    of Arabic. In In Proceedings of LREC, Reykjavik, Iceland, 2014.

(4) N. Habash. Introduction to Arabic Natural Language Processing,
    Synthesis Lectures on Human Language Technologies, Graeme Hirst,
    editor. Morgan & Claypool Publishers. 187 pages, 2010.


====================================================================
Copyright (c) 2015 Columbia University and Carnegie Mellon University
Qatar.  All rights reserved.
====================================================================

