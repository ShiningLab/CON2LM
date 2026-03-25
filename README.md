# CON2LM
This repository is for the paper Word Surprisal Correlates with Sentential Contradiction in LLMs. In *Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 4549–4564, Rabat, Morocco. Association for Computational Linguistics.

[[Paper](https://aclanthology.org/2026.eacl-long.211.pdf)] [[Slides](assets/slides.pdf)]

## Overview
CON2LM investigates how large language models detect contradictions through word-level probability analysis. The key insight is that **word surprisal (negative log probability) correlates with sentence-level contradiction** between a premise and hypothesis.

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
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── config.py              # Configuration management (CLI args, paths, model settings)
├── test.py                # Quick test script for token-to-word algorithm demo
├── run_llms.py            # Main script: compute word surprisals from LLMs
├── main.ipynb             # Analysis notebook: load surprisals, evaluate metrics
├── figure.ipynb           # Visualization notebook: generate paper figures
├── assets/
│   └── slides.pdf         # Conference presentation slides
├── src/
│   ├── helper.py          # Utility functions (device detection, logging, I/O)
│   ├── con2lm.py          # Core CON2LM algorithm (beam search, word probability)
│   └── model_configs.py   # Model-specific settings (BOW prefixes, pad tokens)
└── res/
    ├── data/              # Input TSV datasets (premise, hypothesis, label)
    ├── llms/              # Downloaded language model files
    ├── results/           # Computed surprisals (compressed JSON)
    └── figures/           # Generated visualizations (PDF)
```

## Setup
It is recommended to use a virtual environment to manage dependencies. Follow the steps below to set up the environment and install the required packages:
```sh
$ cd CON2LM
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

## Quick Test
To quickly test the token-to-word decoding algorithm, run:
```sh
$ python test.py
```

**Note:** You need to download a language model first (see Usage section below).

## Usage

### 1. Data Preparation
Place your TSV datasets in `res/data/` with columns: `premise`, `hypothesis`, `label` (boolean). Example datasets are referenced in the paper:
- bAbI: `babi_120_con.tsv`
- SNLI: `snli_1000_con_valid.tsv`, `snli_1000_con_test.tsv`
- Wikipedia: `capital_100_con_v1.tsv`, `capital_100_con_v2.tsv`, `lan_100_con.tsv`, `soft_100_con.tsv`

### 2. Download Language Models
Download models to `res/llms/` or specify paths in `config.py`:
- Llama-3.2-3B / Llama-3.2-3B-Instruct
- gemma-3-4b-pt
- Qwen3-4B

### 3. Compute Word Surprisals
Run `run_llms.py` to compute surprisals for each premise-hypothesis pair:
```sh
$ python run_llms.py --llm Qwen3-4B --temp True --seed 0
```

### 4. Analyze Results
Open `main.ipynb` to:
- Load computed surprisals from `res/results/`
- Apply aggregation strategies (Last/Max/Mean word surprisal)
- Evaluate contradiction detection with ROC-AUC and threshold-based classification
- Compare direct vs. relative surprisal metrics

### 5. Generate Figures
Run `figure.ipynb` to reproduce paper visualizations:
- Threshold tuning curves for different aggregation methods
- Context format comparison (H-only, CAT, TEMP)
- Main results: Accuracy and ROC-AUC bar charts across all datasets

## Authors
* **Ning Shi** - mrshininnnnn@gmail.com

## BibTeX
```bibtex
@inproceedings{shi-etal-2026-word,
    title = "Word Surprisal Correlates with Sentential Contradiction in {LLM}s",
    author = "Shi, Ning  and
      Hauer, Bradley  and
      Basil, David  and
      Zhang, John  and
      Kondrak, Grzegorz",
    editor = "Demberg, Vera  and
      Inui, Kentaro  and
      Marquez, Llu{\'i}s",
    booktitle = "Proceedings of the 19th Conference of the {E}uropean Chapter of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = mar,
    year = "2026",
    address = "Rabat, Morocco",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.eacl-long.211/",
    pages = "4549--4564",
    ISBN = "979-8-89176-380-7",
    abstract = "Large language models (LLMs) continue to achieve impressive performance on reasoning benchmarks, yet it remains unclear how their predictions capture semantic consistency between sentences. We investigate the important open question of whether word-level surprisal correlates with sentence-level contradiction between a premise and a hypothesis. Specifically, we compute surprisal for hypothesis words across a diverse set of experimental variants, and analyze its association with contradiction labels over multiple datasets and open-source LLMs. Because modern LLMs operate on subword tokens and can not directly produce reliable surprisal estimates, we introduce a token-to-word decoding algorithm that extends theoretically grounded probability estimation to open-vocabulary settings. Experiments show a consistent and statistically significant positive correlation between surprisal and contradiction across models and domains. Our analysis also provides new insights into the capabilities and limitations of current LLMs. Together, our findings suggest that surprisal can localize sentence-level inconsistency at the word level, establishing a quantitative link between lexical uncertainty and sentential semantics. We plan to release our code and data upon publication."
}
```
