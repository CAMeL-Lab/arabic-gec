# Character Edit Distance Based Word Alignment

The purpose of this code is to find alignments between parallel sentences 
in the same language but with different spelling conventions.  The ideal use
case is that of raw text with spelling errors, and its corrected form.

## Basic Approach
For each sentence, a basic edit distance based alignment is performed at the word level.
 This is followed by a post-processing step that looks at the context of the edit distance
 operations and decides the best match between the words within the sentence.

## Assumptions
- Sentences are aligned.
- No changes to the word order.
- Text is in the same script and encoding.

#
## Contents
- `align_text.py` main script that produces the alignments.
- `alignmnet.py` basic alignment script that is used in the initial step.
- `requirements.txt` necessary dependencies needed to run the scripts.
- sample/ a directory with sample sentences for demonstrating the examples below.
- `README.md` this document.

## Requirements
- Python 3.5 and above.

To use, you need to first install the necessary dependencies by running the following command:
```
pip install -r requirements.txt
```
#
## Usage
```
Usage:
    align_text (-r RAWSENT | --raw=RAWSENT)
                (-c CORRECTSENT | --correct=CORRECTSENT)
                (-m MODE | --mode=MODE)
                (-o OUTSTR | --out OUTSTR)
                [-h HELP]

Options:
  -r RAWSENT --raw=RAWSENT  RAW sentences file
  -c CORRECTSENT --correct=CORRECTSENT  CORRECT sentences file
  -m MODE --mode=MODE  
        Two modes to choose from: 
            1- 'align' To produce full alignments (one-to-many and many-to-one)
            2- 'basic' To produce basic alignments with operation and distance details (one-to-one)
            [default: align]
  -o OUTSTR --out=OUTSTR  Prefix for single output files
  -h --help  Show this screen.
```
#
## Examples from Arabic (Raw vs Corrected)

### Inputs
### Raw sentence
```
خالد : اممممممممممممممممم اذا بتروحون العصر اوكي ماعندي مانع بس لاتتأخرون
```
### Correct sentence
```
خالد : امم اذا بتروحون العصر اوكيه ما عندي مانع بس لا تتأخرون
```
#
### Full alignment
```
python align_text.py -r sample/sample.ar.raw -c sample/sample.ar.correct -m align -o sample/sample.ar
```

### Output
```
RAW alignments are saved to: sample/sample.ar.raw.align
CORRECT alignments are saved to: sample/sample.ar.correct.align
Side by side alignments are saved to: sample/sample.ar.coAlign
```
### Side by side view (found in the _.coAlign_ file)

|RAW| CORRECT|
|---|-----|
|خالد | خالد |
|: | : |
|اممممممممممممممممم | امم |
|اذا | اذا |
|بتروحون | بتروحون |
|العصر | العصر |
|اوكي | اوكيه |
|ماعندي | ما عندي |
|مانع | مانع |
|بس | بس |
|لاتتأخرون | لا تتأخرون |

### Notes on output
You can notice here whenever there is a _split_ or _merge_ on either side they are collapsed on the respective side, thus, we can have one-to-many, and many-to-one cases.

#

### Basic alignment
```
python align_text.py -r sample/sample.ar.raw -c sample/sample.ar.correct -m basic -o sample/sample.ar
```
### Output
```
Basic alignments are saved to: sample/sample.ar.basic
```
|RAW|op|CORRECT|Alignment Details|
|- |- |- |- |
|خالد| =| خالد| (1, 1, 'n', 0)|
|:| =| :| (2, 2, 'n', 0)|
|اممممممممممممممممم| \|| امم| (3, 3, 's', 1.7)|
|اذا| =| اذا| (4, 4, 'n', 1.7)|
|بتروحون| =| بتروحون| (5, 5, 'n', 1.7)|
|العصر| =| العصر| (6, 6, 'n', 1.7)|
|اوكي| \|| اوكيه| (7, 7, 's', 2.1)|
| |<| ما|(None, 8, 'i', 3.1)|
|ماعندي| \|| عندي| (8, 9, 's', 3.7)|
|مانع| =| مانع| (9, 10, 'n', 3.7)|
|بس| =| بس| (10, 11, 'n', 3.7)|
| |<| لا|(None, 12, 'i', 4.7)|
|لاتتأخرون| \|| تتأخرون| (11, 13, 's', 5.2)|

### Notes on output
- Operations (op) are defined as follows:

|op|Description|
|-|-|
|=|  No change (n)|
|\||  Substitution (s)|
|< | Insertion (i)|
|> | Deletion (d)|

- Alignment Details is a compact representation of the alignment:
```
(<source_idx>, <targe_idx>, op, editdistance_score)
```
