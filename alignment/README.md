# Grammatical Error Correction Alignment:
The [M2 Scorer](https://github.com/nusnlp/m2scorer) has well known issues when it comes to generating M2 files. It tends to cluster multiple tokens together in a single edit and this leads to penalizing models even if they generate partially correct answers. The tool in this repo introduces a flexible alignment algorithm to align words between two parallel sentences, where the source sentence has spelling and grammatical errors and the target sentence has the corrections.

The algorithm handles all alignment types: insertions, deletions, replacements, splits, and merges. For instance, given the following two sentences:

Source:
```
خالد : اممممممممممممممممم اذا بتروحووون العصر الساعه ٢ اوكي ماعندي مانع لاتتأخرون و كلمو احمد هالمر ة لوسمحتم
```

Target:
```
خالد ، اذا بتروحون العصر الساعة 2 اوكيه ما عندي مانع بس لا تتأخرون وكلمو أحمد هذه المرة لو سمحتم .
```

The algorithm would generate the following alignment:

|Source|Target|
|--------------------|----------------------------|
|خالد | خالد |
| : |  ،|
| اممممممممممممممممم| |
|اذا | اذا |
|بتروحووون | بتروحون |
|العصر | العصر |
|الساعه | الساعة |
| ٢ | 2 |
| اوكي  | اوكيه |
|ماعندي | ما عندي |
|مانع | مانع |
| | بس |
|لاتتأخرون | لا تتأخرون |
| و كلمو | وكلمو |
|احمد | أحمد |
| هالمر ة | هذه المرة |
| لوسمحتم | لو سمحتم |
| | . |

## Generating Alignment:

```python
python aligner.py --src /path/to/src --tgt /path/to/tgt --output /path/to/output
```

To run the alignment on the sample files:

```python
python aligner.py --src sample/src.txt --tgt sample/tgt.txt --output sample/alignment.txt
```

## Generating M2 Files:

After generating the word-level alignment, we provide scripts to generate the M2 file that could be used with the M2 scorer for evaluation

```python 
python create_m2_file.py --src /path/to/src --tgt /path/to/tgt 
                         --align /path/to/alignment --output /path/to/output
```

To run this script on the provided sample files:

```python
python create_m2_file.py --src sample/src.txt --tgt sample/tgt.txt 
                         --align sample/alignment.txt --output sample/edits.m2
```


