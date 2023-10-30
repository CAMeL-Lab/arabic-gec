# Arabic Grammatical Error Detection and Correction

This repo contains code and pretrained models to reproduce the results in our paper [Advancements in Arabic Grammatical Error Detection and Correction: An Empirical Investigation](https://arxiv.org/abs/2305.14734).

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

conda create -n python=3.7
pip install -r requirements.txt
```

## Experiments and Reproducibility:

This repo is organized as follows:
1. [data]():

2. [ged]():

3. [gec]():

4. [alignment]():

5. [areta]():

6. [transformers]():

7. [utils]():

8. [stat_significance]():
    

## License:

This repo is available under the MIT license. See the [LICENSE](LICENSE) for more info.


## Citation:

If you find the code or data in this repo helpful, please cite [our paper](https://arxiv.org/abs/2305.14734):

```bibtex
@inproceedings{alhafni-etal-2023-advancements,
    title = "Advancements in Arabic Grammatical Error Detection and Correction: An Empirical Investigation",
    author = "Alhafni, Bashar and
      Inoue, Go and
      Khairallah, Christian  and
      Habash, Nizar",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore, Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2305.14734",
    abstract = "Grammatical error correction (GEC) is a well-explored problem in English with many existing models and datasets. However, research on GEC in morphologically rich languages has been limited due to challenges such as data scarcity and language complexity. In this paper, we present the first results on Arabic GEC using two newly developed Transformer-based pretrained sequence-to-sequence models. We also define the task of multi-class Arabic grammatical error detection (GED) and present the first results on multi-class Arabic GED. We show that using GED information as auxiliary input in GEC models improves GEC performance across three datasets spanning different genres. Moreover, we also investigate the use of contextual morphological preprocessing in aiding GEC systems. Our models achieve SOTA results on two Arabic GEC shared task datasets and establish a strong benchmark on a recently created dataset. We make our code, data, and pretrained models publicly available.",
}
