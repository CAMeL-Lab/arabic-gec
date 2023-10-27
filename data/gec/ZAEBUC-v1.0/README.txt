================================================================
                            ZAEBUC
    The Zayed Arabic-English Bilingual Undergraduate Corpus

                            Release #1
                           16 June 2022

=================================================================
Summary
=================================================================

The Zayed Arabic-English Bilingual Undergraduate Corpus (ZAEBUC) is an
annotated Arabic-English writer corpus comprising short essays written
by first-year students at Zayed University in the United Arab Emirates.
This dataset is an open, publicly available and extendable resource,
designed with the intention to support empirically driven research in
Arabic, English, and bilingual development, as well as research and
system development in natural language processing (NLP).

ZAEBUC matches comparable texts in different languages written by the
same writer on different occasions.  The corpus consists of 388 English
essays (87.6K Raw tokens), and 214 Arabic essays(33.3K Raw tokens),
written by a total of 397 students -- 52% of participating students
wrote texts in both languages, and 73% of those students wrote on the
same topic.

The corpus is enriched with the following annotations:
(a) anonymized meta-data indicating
	extra-linguistic features of the writers and texts; 
(b) a manually corrected version of the raw text; 
(c) automatic and manual annotations to identify morphological tokens,
	part-of-speech (POS), and lemmas; and
(d) writing proficiency ratings using the Common European Framework of
	Reference (CEFR).


When citing this resource, please use:

Nizar Habash & David Palfreyman. ZAEBUC: An Annotated Arabic-English
Bilingual Writer Corpus. In Proceedings of the 13th Conference on
Language Resources and Evaluation (LREC 2022), pages 79-88, Marseille,
2022.

@inproceedings{habash-palfreyman:2022:LREC,
	abstract = {We present ZAEBUC, an annotated Arabic-English bilingual writer corpus comprising short essays by first-year university students at Zayed University in the United Arab Emirates. We describe and discuss the various guidelines and pipeline processes we followed to create the annotations and quality check them. The annotations include spelling and grammar correction, morphological tokenization, Part-of-Speech tagging, lemmatization, and Common European Framework of Reference (CEFR) ratings. All of the annotations are done on Arabic and English texts using consistent guidelines as much as possible, with tracked alignments among the different annotations, and to the original raw texts. For morphological tokenization, POS tagging, and lemmatization, we use existing automatic annotation tools followed by manual correction. We also present various measurements and correlations with preliminary insights drawn from the data and annotations. The publicly available ZAEBUC corpus and its annotations are intended to be the stepping stones for additional annotations.},
	address = {Marseille, France},
	author = {Habash, Nizar and Palfreyman, David},
	booktitle = {Proceedings of the Language Resources and Evaluation Conference},
	month = {June},
	pages = {79--88},
	publisher = {European Language Resources Association},
	title = {ZAEBUC: An Annotated Arabic-English Bilingual Writer Corpus},
	url = {https://aclanthology.org/2022.lrec-1.9},
	year = {2022},
	Bdsk-Url-1 = {https://aclanthology.org/2022.lrec-1.9}}

=================================================================
Description of Files
=================================================================

The zipped folder "ZAEBUC-v1.0.zip" has the following contents:

(1) README.txt  :   This file.

(2) LICENSE.txt :   The license to use this lexicon.

(3) 2022.lrec-1.9-Habash-Palfreyman.pdf :
                The main paper describing the resource.

(4) AR-all.alignment-FINAL.tsv 
& 
(5) EN-all.alignment-FINAL.tsv:
	Spelling correction files, containing aligned Raw-to-Corrected
	tokens, along with a description of each edit operation.
	
(6) AR-all.extracted.corrected.analyzed.corrected-FINAL.tsv 
&
(7) EN-all.extracted.corrected.analyzed.corrected-FINAL.tsv: 
	The corrected tokens enriched with multiple layers of automatic and
	manual annotations.  

=================================================================
Description of aligned spelling correction files.
=================================================================

Both files contain the same 4 tab separated fields:

(1) Document: the document ID, for instance "EN-140-116710", where "EN"
				is the language of the document, "140" is the ID of the
				course associated with the essay submission, and
				"116710" is the anonymized author ID.

(2) Raw: the raw input as submitted by the students.

(3) Corrected: the corrected version of the raw input.

(4) Operation: 	the operation required to edit each raw token to its
				corrected version.

=================================================================
Description of the corrected and annotated files.
=================================================================

Both files share the first 10 columns, and contain additional language
specific columns.

Shared columns:

(1) Document: the document ID, (similar to the previous tsvs).

(2) Line_Index: the index of the line that the token belongs to.

(3) Word: the spelling corrected word.

(4) Flag: 	correction annotators were instructed not to correct
			open-class lexical choices.  This column marks open-class
			cases deemed wrong by annotators with an "!", or suggests
			more appropriate alternatives.

(5) Auto_Tokenization & 
(6) Auto_POS:	automatically generated, following Universal Dependency
				guidelines, i.e. PTB tokenization for English and PATB
				for Arabic, and UD parts-of-speech.

(7) Auto_Lemma: An automatically generated abstraction that represents
				the various inflectional forms of a particular lexical
				item with a specific derivation and POS.

(8) Manual_Tokenization & 
(9) Manual_POS &
(10) Manual_Lemma:	Manually corrected versions of Auto_Tokenization,
					Auto_POS, and Manual_Lemma.


Additional Arabic columns:

(11) Manual_Diacritized_Lemma: the manually diacritized version of the
								lemmas. This annotation was finished in
								a later phase of the project, after the
								finalization of the paper describing the
								corpus. As such it is not included in
								the paper.

(12) Gloss: A list of english glosses associated with the word.


================================================================
Acknowledgments
================================================================

The creators of this corpus acknowledge with appreciation the support of
this project from the Zayed University Research Incentive Fund (RIF
#R19068).

We also extend thanks to Ramy Eskander for helpful discussions and the
team of annotators at Ramitechs for their help in creating this
resource.

================================================================
References
================================================================

Council of Europe (2001). Common European Framework of Reference for
Languages: learning, teaching, assessment. Cambridge University Press.
 
Grosjean, F. (2010). Bilingual. Harvard University Press.
 
Kilgarriff, A., Baisa, V., Bušta, J., Jakubíček, M., Kovář, V.,
Michelfeit, J., ... & Suchomel, V. (2014). The Sketch Engine: ten years
on. Lexicography, 1(1), 7-36.
 
Maamouri, M., Bies, A., Buckwalter, T., & Mekki, W. (2004, September).
The Penn Arabic Treebank: Building a large-scale annotated arabic
corpus. In NEMLAR conference on Arabic language resources and tools
(Vol. 27, pp. 466-467).
 
Marcus, M., Santorini, B., & Marcinkiewicz, M. A. (1993). Building a
large annotated corpus of English: The Penn Treebank.
 
Nivre, Joakim, Marie-Catherine De Marneffe, Filip Ginter, Yoav Goldberg,
Jan Hajic, Christopher D. Manning, Ryan McDonald et al. "Universal
dependencies v1: A multilingual treebank collection." In Proceedings of
the Tenth International Conference on Language Resources and Evaluation
(LREC'16), pp. 1659-1666. 2016.
 
Siemund, P., Al‐Issa, A., & Leimgruber, J. R. (2020). Multilingualism
and the role of English in the United Arab Emirates. World Englishes.
https://onlinelibrary.wiley.com/doi/pdf/10.1111/weng.12507
 
UNESCO (2016). United Arab Emirates. http://uis.unesco.org/en/country/ae

================================================================
Copyright (c) 2022 David Palfreyman and Nizar Habash
================================================================
