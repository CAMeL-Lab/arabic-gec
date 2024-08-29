# Arabic Grammatical Error Detection and Correction

This repo contains code and pretrained models to reproduce the results in our paper [Advancements in Arabic Grammatical Error Detection and Correction: An Empirical Investigation](https://aclanthology.org/2023.emnlp-main.396.pdf).

## Requirements:

The code was written for python>=3.9, pytorch 1.11.1, and a modified version of transformers 4.22.2. You will need a few additional packages. Here's how you can set up the environment using conda (assuming you have conda and cuda installed):

```bash
git clone https://github.com/CAMeL-Lab/arabic-gec.git
cd arabic-gec

conda create -n gec python=3.9
conda activate gec

pip install -r requirements.txt
```

To obtain multi-class grammatical error detection (GED) labels, we use an enhanced version of [ARETA](https://arxiv.org/abs/2109.08068). To avoid dependency conflicts and to ensure reproducibility, you would need to create a separate environment to run ARETA:

```bash
cd areta

conda create -n areta python=3.7
pip install -r requirements.txt
```

## Experiments and Reproducibility:

This repo is organized as follows:
1. [data](data): includes all the data we used throughout our paper to train and test various systems. This includes alignments, m2edits, GED, GEC, and all the utilities we used.
2. [ged](ged): includes the scripts needed to train and evaluate our GED models.
3. [gec](gec): includes the scripts needed to train and evaluate our GEC models.
4. [alignment](alignment): a stand-alone script for the alignment algorithm we introduced in our paper. 
5. [areta](areta): a slightly improved version of ARETA, an error type annotation tool for Modern Standard Arabic (MSA).
6. [transformers](transformers): our extended version of Hugging Face's transformers that allows incorporating GED information with seq2seq models.


## Hugging Face Integration:

We make our GED and GEC models publicly available on [Hugging Face](https://huggingface.co/collections/CAMeL-Lab/arabic-ged-and-gec-6541e5996be058da06556994).

### GED:
```python

from transformers import pipeline
ged = pipeline('token-classification', model='CAMeL-Lab/camelbert-msa-qalb14-ged-13')
text = 'و قال له انه يحب اكل الطعام بكثره'
predictions = ged(text)
print(predictions)

"""[{'entity': 'MERGE-B', 'score': 0.99943775, 'index': 1, 'word': 'و', 'start': 0, 'end': 1}, {'entity': 'MERGE-I', 'score': 0.99959165, 'index': 2, 'word': 'قال', 'start': 2, 'end': 5}, {'entity': 'UC', 'score': 0.9985884, 'index': 3, 'word': 'له', 'start': 6, 'end': 8}, {'entity': 'REPLACE_O', 'score': 0.8346316, 'index': 4, 'word': 'انه', 'start': 9, 'end': 12}, {'entity': 'UC', 'score': 0.99985325, 'index': 5, 'word': 'يحب', 'start': 13, 'end': 16}, {'entity': 'REPLACE_O', 'score': 0.6836415, 'index': 6, 'word': 'اكل', 'start': 17, 'end': 20}, {'entity': 'UC', 'score': 0.99763715, 'index': 7, 'word': 'الطعام', 'start': 21, 'end': 27}, {'entity': 'REPLACE_O', 'score': 0.993848, 'index': 8, 'word': 'بكثره', 'start': 28, 'end': 33}]"""
```

### GEC:
```python

from transformers import AutoTokenizer, BertForTokenClassification, MBartForConditionalGeneration
from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
from camel_tools.utils.dediac import dediac_ar
import torch.nn.functional as F
import torch

bert_disambig = BERTUnfactoredDisambiguator.pretrained()

ged_tokenizer = AutoTokenizer.from_pretrained('CAMeL-Lab/camelbert-msa-qalb14-ged-13')
ged_model = BertForTokenClassification.from_pretrained('CAMeL-Lab/camelbert-msa-qalb14-ged-13')

gec_tokenizer = AutoTokenizer.from_pretrained('CAMeL-Lab/arabart-qalb14-gec-ged-13')
gec_model = MBartForConditionalGeneration.from_pretrained('CAMeL-Lab/arabart-qalb14-gec-ged-13')

text = 'و قال له انه يحب اكل الطعام بكثره .'

# morph processing the input text
text_disambig = bert_disambig.disambiguate(text.split())
morph_pp_text = [dediac_ar(w_disambig.analyses[0].analysis['diac']) for w_disambig in text_disambig]
morph_pp_text = ' '.join(morph_pp_text)

# GED tagging
inputs = ged_tokenizer([morph_pp_text], return_tensors='pt')
logits = ged_model(**inputs).logits
preds = F.softmax(logits, dim=-1).squeeze()[1:-1]
pred_ged_labels = [ged_model.config.id2label[p.item()] for p in torch.argmax(preds, -1)]

# Extending GED label to GEC-tokenized input
ged_label2ids = gec_model.config.ged_label2id
tokens, ged_labels = [], []

for word, label in zip(morph_pp_text.split(), pred_ged_labels):
    word_tokens = gec_tokenizer.tokenize(word)
    if len(word_tokens) > 0:
         tokens.extend(word_tokens)
         ged_labels.extend([label for _ in range(len(word_tokens))])


input_ids = gec_tokenizer.convert_tokens_to_ids(tokens)
input_ids = [gec_tokenizer.bos_token_id] + input_ids + [gec_tokenizer.eos_token_id]

label_ids = [ged_label2ids.get(label, ged_label2ids['<pad>']) for label in ged_labels]
label_ids = [ged_label2ids['UC']] + label_ids + [ged_label2ids['UC']]
attention_mask = [1 for _ in range(len(input_ids))]


gen_kwargs = {'num_beams': 5, 'max_length': 100,
              'num_return_sequences': 1,
              'no_repeat_ngram_size': 0, 'early_stopping': False,
              'ged_tags': torch.tensor([label_ids]),
              'attention_mask': torch.tensor([attention_mask])
              }

# GEC generation
generated = gec_model.generate(torch.tensor([input_ids]), **gen_kwargs)
generated_text = gec_tokenizer.batch_decode(generated,
                                            skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False
                                            )[0]
print(generated_text)
""" وقال له أنه يحب أكل الطعام بكثرة ."""
```


## License:

This repo is available under the MIT license. See the [LICENSE](LICENSE) for more info.


## Citation:

If you find the code or data in this repo helpful, please cite [our paper](https://aclanthology.org/2023.emnlp-main.396.pdf):

```bibtex
@inproceedings{alhafni-etal-2023-advancements,
    title = "Advancements in {A}rabic Grammatical Error Detection and Correction: An Empirical Investigation",
    author = "Alhafni, Bashar  and
      Inoue, Go  and
      Khairallah, Christian  and
      Habash, Nizar",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.396",
    pages = "6430--6448",
    abstract = "Grammatical error correction (GEC) is a well-explored problem in English with many existing models and datasets. However, research on GEC in morphologically rich languages has been limited due to challenges such as data scarcity and language complexity. In this paper, we present the first results on Arabic GEC using two newly developed Transformer-based pretrained sequence-to-sequence models. We also define the task of multi-class Arabic grammatical error detection (GED) and present the first results on multi-class Arabic GED. We show that using GED information as auxiliary input in GEC models improves GEC performance across three datasets spanning different genres. Moreover, we also investigate the use of contextual morphological preprocessing in aiding GEC systems. Our models achieve SOTA results on two Arabic GEC shared task datasets and establish a strong benchmark on a recently created dataset. We make our code, data, and pretrained models publicly available.",
}
