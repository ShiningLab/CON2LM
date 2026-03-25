#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Ning Shi'
__email__ = 'mrshininnnnn@gmail.com'

"""
Model-specific configurations and mappings.
"""

# Global dictionary mapping model IDs (or model families) to BOW prefixes
BOW_PREFIX_MAP = {
    'HF/Llama-3.2-3B': 'Ġ'
    , 'HF/Llama-3.2-3B-Instruct': 'Ġ'
    , 'HF/gemma-3-4b-pt': '▁'
    , 'HF/Qwen3-4B': 'Ġ'
    , 'HF/DeepSeek-R1-Distill-Qwen-1.5B': 'Ġ'
}

# Pad token mappings for models that need special handling
PAD_TOKEN_MAP = {
    'HF/Qwen3-4B': '<|endoftext|>'
}

# Default BOW prefix if model not found in mapping
DEFAULT_BOW_PREFIX = 'Ġ'