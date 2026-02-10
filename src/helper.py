#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Ning Shi'
__email__ = 'mrshininnnnn@gmail.com'


# built-in
import os
import json
import gzip
import logging
import random
# public
import torch
import numpy as np
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


def get_device():
    if "DEVICE" in os.environ:
        return os.environ["DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def set_random_seed(seed: int = 42):
    """Fix random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # Ensures deterministic behavior where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_logger(log_path):
    """Initialize logger for experiment tracking."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    return logger


def save_json_gzip(data, file_path):
    """Save data as compressed JSON file."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with gzip.open(file_path, 'wt', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json_gzip(file_path):
    """Load data from compressed JSON file."""
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        return json.load(f)

def is_content_word(word, pos):
    """Check if word is a content word based on POS tag"""
    content_pos = {'NN', 'NNS', 'NNP', 'NNPS',  # Nouns
                    'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',  # Verbs
                    'JJ', 'JJR', 'JJS',  # Adjectives
                    'RB', 'RBR', 'RBS'}  # Adverbs
    return pos in content_pos and word.lower() not in stop_words