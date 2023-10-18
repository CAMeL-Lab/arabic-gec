# Arabic Error Type Annotation

## Description
The Arabic Error Type Annotation tool (ARETA) aims to annotate Arabic error types following the Arabic Learner Corpus ([ALC](https://www.arabiclearnercorpus.com/)) tagset annotation. ARETA is described in [Automatic Error Type Annotation for Arabic (Belkebir and Habash, 2021)](https://aclanthology.org/2021.conll-1.47.pdf).
## Installation
You will need Python 3.7 and above (64-bit).

1. Install [CamelTools](https://github.com/CAMeL-Lab/camel_tools#install-using-pip).
2. Install requirements.
```
pip install -r requirements.txt
```

## Usage
### Error Type Annotation:
```
Usage: annotate_err_type_ar.py [OPTIONS] --sys_path system --ref_path reference 
where
    system - the system output
    reference - the reference file
OPTIONS
    --show_edit_paths - whether to show the shortest edit paths or not; defaults to false.
    --output_path - output file directory; defaults to standard output.
    
```

Example:

```
python annotate_err_type_ar.py --sys_path sample/sys_sample.txt --ref_path sample/ref_sample.txt
```

The output lists triplets of system, reference and error types. For the complete list of error types, see [table of error types](#table-of-error-types) below.

Example:

System sentence:  إن أمتحان الاستاذة صعبة

Reference sentence: إن إمتحان الأستاذ صعب

Annotation result:

|      System    |   Reference      |  Error type     | 
|----------|---------|-------| 
| إن       | إن      | UC    | 
| أمتحان   | إمتحان  | OH    | 
| الاستاذة | الأستاذ | OH+XG | 
| صعبة     | صعب     | XG    | 

 
**Note**: It is important to note that every sentence in the system/reference outputs must start with an `s`. Refer to [sample/sys_sample.txt](https://github.com/CAMeL-Lab/arabic_error_type_annotation/tree/main/sample/sys_sample.txt) and [sample/ref_sample.txt](https://github.com/CAMeL-Lab/arabic_error_type_annotation/tree/main/sample/ref_sample.txt) for examples on how these files should look like.


### Annotation and evaluation using m2 files (Command Line):
```
Usage: annotate_eval_ar.py [OPTIONS] system source_reference 
where
    system - the system output
    source_reference - source sentences with gold token edits (.m2 file)
OPTIONS
```

Example:

```
python annotate_eval_ar.py sample/CLMB-1 sample/QALB-Test2014.m2
```

This generates:
1. ```annot_input_ref.tsv``` file in the ```output``` folder that contains  the error types annotation between the input and the reference.
2. ```annot_input_sys.tsv``` file in the ```output``` folder that contains the error types annotation between the input and the system.
3.  ```subclasses_results_CLMB-1.tsv``` file in the ```results``` folder. This file contains the results of the evaluation of the system's output against the reference.

## Utilities
### Generate source from m2 file
```
Usage: utilities/generate-m2-source.py m2-file
where
    m2-file -   the m2 file
```

Example:

```python utilities/generate-m2-source.py sample/QALB-Test2014.m2  > sample/QALB-Test2014.source.sent```

### Generate reference from m2 file
```
Usage: utilities/generate-m2-reference.py m2-file
where
    m2-file -   the m2 file
```

Example:

```python utilities/generate-m2-reference.py sample/QALB-Test2014.m2 > sample/QALB-Test2014.reference.cor```


### Adjust the alignment

This re_alignment tool realigns files from Ossama's basic aligner by shifting the word in null -> word pair to the word before or after according to minimum edit distance.  
```
Usage: utilities/adjust_align_tool.py file_to_adjust_align
where
    file_to_adjust_align -   File to be realigned (should follow Ossama's basic alignement file format)
```

Example:

```python utilities/adjust_align_tool.py sample/align.basic  > sample/align.adjust```

## Configuration
In the configuration file ```config.json```, the user should specify the mode of the morphological analyser. The default value is ```analyser``` in which all the analyses are considered. The second option is 
```mle``` and in this case, we need to specify the parameter ```mle_top``` which represents the maximum number of analyses to be considered. The ```uc``` parameter takes the values 0 or 1. 0 to indicate that we do not consider the unchanged error types, 1 otherwise.  

```
{
  "mode": "analyser",
  "mle_top": ""
}
```

## Table of Error Types

|               |           |                                                     |                            |                            | 
|---------------|-----------|-----------------------------------------------------|----------------------------|----------------------------| 
| **Class**         | **Sub-class** | **Description**                                         | **Arabic Example**             | **Buckwalter Transliteration** | 
| **Orthographic**  | OH        | Hamza error                                         | اكثر← أكثر                 | Akvr → >kvr                | 
|               | OT        | Confusion in Ha and Ta Mutadarrifatin               | مشاركه ← مشاركة            | m$Arkh → m$Arkp            | 
|               | OA        | Confusuion in Alif and Ya Mutadarrifatin            | علي ← على                  | Ely → ElY                  | 
|               | OW        | Confusion in Alif Fariqa                            | وكانو ←  وكانوا            | wkAnw→ wkAnwA              | 
|               | ON        | Confusion Between Nun and Tanwin                    | ثوبن ← ثوبٌ                | vwbn → vwbN                | 
|               | OS        | Shortening the long vowels                          | أوقت ← أوقات               | >wqt → >wqAt               | 
|               | OG        | Lengthening the short vowels                        | نقيمو ← نقيم               | nqymw → nqym               | 
|               | OC        | Wrong order of word characters                      | تبرينا ← تربينا            | tbrynA → trbynA            | 
|               | OR        | Replacement in word character(s)                    | مصلنا ← وصلنا              | mSlnA → wSlnA              | 
|               | OD        | Additional character(s)                             | يعدوم ← يدوم               | yEdwm → ydwm               | 
|               | OM        | Missing character(s)                                | سالين ← سائلين             | sAlyn → sA}lyn             | 
|               | OO        | Other orthographic errors                           | -                          | -                          | 
| **Morphological** | MI        | Word inflection                                     | معروف ← عارف               | mErwf → EArf               | 
|               | MT        | Verb tense                                          | تفرحني ← أفرحتني           | tfrHny → >frHtny           | 
|               | MO        | Other morphological errors                          | -                          | -                          | 
| **Syntax**        | XC        | Case                                                | رائع ← رائعاً              | rA}E → rA}EAF              | 
|               | XF        | Definiteness                                        | السن ← سن                  | Alsn → sn                  | 
|               | XG        | Gender                                              | الغربي ← الغربية           | Algrby → Algrbyp           | 
|               | XN        | Number                                              | فكرتي ← أفكاري             | fkrty → >fkAry             | 
|               | XT        | Unnecessary word                                    | على ←Null                  | ElY →Null                  | 
|               | XM        | Missing word                                        | Null ← على                 | Null → ElY                 | 
|               | XO        | Other syntactic errors                              | -                          | -                          | 
| **Semantic**      | SW        | Word selection error                                | من ← عن                    | mn → En                    | 
|               | SF        | Fasl wa wasl (confusion in conjunction use/non-use) | سبحان ← فسبحان             | sbHAn → fsbHAn             | 
|               | SO        | Other semantic errors                               | -                          | -                          | 
| **Punctuation**   | PC        | Punctuation confusion                               | المتوسط. ← المتوسط،        | AlmtwsT. → AlmtwsT،        | 
|               | PT        | Unnecessary punctuation                             | العام,  ← العام            | AlEAm,  → AlEAm            | 
|               | PM        | Missing punctuation                                 | العظيم ←  العظيم،          | AlEZym →  AlEZym،          | 
|               | PO        | Other errors in punctuation                         | -                          | -                          | 
|               |           |                                                     |                            |                            | 
| **Merge**         | MG        | Words are merged                                    | ذهبتالبارحة ← ذهبت البارحة | *hbtAlbArHp → *hbt AlbArHp | 
| **Split**         | SP        | Words are split                                     | المحا دثات ← المحادثات     | AlmHA dvAt → AlmHAdvAt     | 

## Citation
If you find ARETA useful in your research, please cite
**Automatic Error Type Annotation for Arabic (Belkebir and Habash, 2021)** ([PDF](https://aclanthology.org/2021.conll-1.47.pdf)) ([BIB](https://aclanthology.org/2021.conll-1.47.bib)).

## License
This tool is available under the MIT license. See the [LICENSE file](https://github.com/CAMeL-Lab/arabic_error_type_annotation/blob/main/LICENSE) for more info.

## Contributors
* [Riadh Belkebir](https://github.com/riadhb88)
* [Nizar Habash](https://github.com/nizarhabash1)


