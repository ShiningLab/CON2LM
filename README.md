# CON2LM
This repository is for the paper Word Surprisal Correlates with Sentential Contradiction in LLMs. In *Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics* (EACL 2026). Association for Computational Linguistics.

[[Paper](https://github.com/ShiningLab/CON2LM/blob/main/assets/paper.pdf)] [[Poster](#)] [[Slides](#)]

## Dependencies
Ensure you have the following dependencies installed:
+ python >= 3.11
+ torch >= 2.7.0
+ transformers >= 4.51.0
+ numpy >= 1.26.0
+ pandas >= 2.2.0
+ nltk >= 3.9.0
+ scikit-learn >= 1.5.0
+ matplotlib >= 3.9.0
+ spacy >= 3.7.0

## Directory
```
CON2LM/
├── README.md
├── requirements.txt
├── assets/
│   └── paper.pdf
├── config.py
├── src/
│   └── helper.py
└── res/
    ├── data/
    ├── llms/
    └── results/
```

## Setup
It is recommended to use a virtual environment to manage dependencies. Follow the steps below to set up the environment and install the required packages:
```sh
$ cd CON2LM
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

## Run
The main analysis is implemented in Jupyter notebooks. Review and modify configurations in `config.py` as needed:
```sh
$ vim config.py
$ jupyter notebook main.ipynb
```

## Authors
* **Ning Shi** - mrshininnnnn@gmail.com

## BibTeX
```bibtex
@inproceedings{shi2026con2lm,
  title={Word Surprisal Correlates with Sentential Contradiction in LLMs}, 
  author = "Shi, Ning  and 
  Hauer, Bradley  and 
  Basil, David  and 
  Zhang John  and 
  Kondrak, Grzegorz",
  booktitle={Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics},
  year={2026},
  organization={Association for Computational Linguistics}
}
```
