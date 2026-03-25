#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
GPU-optimized script for computing word-level surprisals for contradiction detection.

This script processes premise-hypothesis pairs from TSV datasets and computes:
1. Surprisal of each word in hypothesis with/without premise context
2. Surprisal of most likely next words at each position with/without premise context

The results are saved as compressed JSON files for further analysis.
"""

__author__ = 'Ning Shi'
__email__ = 'mrshininnnnn@gmail.com'

# Built-in imports
import os
import fire
import pandas as pd

# Public library imports
import torch
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Private module imports
from config import Config
from src import helper
from src.helper import set_random_seed, get_device
from src.model_configs import BOW_PREFIX_MAP, DEFAULT_BOW_PREFIX
from src.con2lm import *

# Template for premise context formatting
CONTEXT_TEMPLATE = 'Since {}, therefore'


def setup_model_and_tokenizer(model_path, model_name, device):
    """
    Load model and tokenizer on specified device with appropriate configurations.

    Args:
        model_path (str): Local path to model files
        model_name (str): Model identifier for BOW prefix lookup (e.g., 'HF/Qwen3-4B')
        device (str): Device to load model on ('cuda', 'cpu', 'mps', etc.)

    Returns:
        tuple: (model, tokenizer, bow_prefix, bow_prefix_id)
    """
    print(f"Loading model {model_name} from {model_path} on {device}...")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device
    )

    # Set model to evaluation mode
    model.eval()

    # Setup BOW (Beginning-of-Word) tokens for word boundary detection
    bow_prefix = BOW_PREFIX_MAP.get(model_name, DEFAULT_BOW_PREFIX)
    bow_prefix_id = tokenizer.convert_tokens_to_ids(bow_prefix)

    return model, tokenizer, bow_prefix, bow_prefix_id


def compute_surprisals_given_words(premise, hypothesis, model, tokenizer, detokenizer, config):
    """
    Compute surprisals for each word in hypothesis using existing beam search functions.

    This function calculates how surprising each word in the hypothesis is when:
    1. The premise is provided as context (with template if config.temp=True)
    2. No premise context is provided (hypothesis only)

    Args:
        premise (str): Premise sentence
        hypothesis (str): Hypothesis sentence
        model: Loaded language model
        tokenizer: Model tokenizer
        detokenizer: NLTK detokenizer for reconstructing text
        config: Configuration object with model settings

    Returns:
        tuple: (surprisals_with_premise, surprisals_without_premise)
            Both are lists of float values, one per word in hypothesis
    """
    # Format premise context using template if enabled
    context = CONTEXT_TEMPLATE.format(premise[:-1]) if config.temp else premise

    # Compute surprisals with premise as context
    beams_with_p = get_sentence_beams(hypothesis, tokenizer, detokenizer, model, config, context=context)
    surprisals_with_p = [-beam.log_prob() for beam in beams_with_p]

    # Compute surprisals without premise (empty context)
    beams_without_p = get_sentence_beams(hypothesis, tokenizer, detokenizer, model, config, context='')
    surprisals_without_p = [-beam.log_prob() for beam in beams_without_p]

    return surprisals_with_p, surprisals_without_p


def compute_surprisals_next_words(premise, hypothesis, model, tokenizer, detokenizer, config):
    """
    Compute surprisals for the most likely next word at each position in hypothesis.

    This function determines what word the model would predict next at each position
    in the hypothesis, and calculates the surprisal of those predictions with/without
    premise context.

    Args:
        premise (str): Premise sentence
        hypothesis (str): Hypothesis sentence
        model: Loaded language model
        tokenizer: Model tokenizer
        detokenizer: NLTK detokenizer for reconstructing text
        config: Configuration object with model settings

    Returns:
        dict: Contains four lists, each with one element per word position:
            - 'next_words_with_premise': Most likely next words with premise context
            - 'next_surprisals_with_premise': Their surprisals
            - 'next_words_without_premise': Most likely next words without premise context
            - 'next_surprisals_without_premise': Their surprisals
    """
    # Format premise context using template if enabled
    context = CONTEXT_TEMPLATE.format(premise[:-1]) if config.temp else premise

    # Get most likely next words at each position with premise context
    next_beams_with_p = get_sentence_topk_beams(hypothesis, tokenizer, detokenizer, model, config, context=context)
    next_words_with_p = [beam.decoded(tokenizer).strip() for beam in next_beams_with_p]
    next_surprisals_with_p = [-beam.log_prob() for beam in next_beams_with_p]

    # Get most likely next words at each position without premise context
    next_beams_without_p = get_sentence_topk_beams(hypothesis, tokenizer, detokenizer, model, config, context='')
    next_words_without_p = [beam.decoded(tokenizer).strip() for beam in next_beams_without_p]
    next_surprisals_without_p = [-beam.log_prob() for beam in next_beams_without_p]

    return {
        'next_words_with_premise': next_words_with_p,
        'next_surprisals_with_premise': next_surprisals_with_p,
        'next_words_without_premise': next_words_without_p,
        'next_surprisals_without_premise': next_surprisals_without_p
    }


def process_dataset(input_file, output_file, model, tokenizer, detokenizer, config, logger):
    """
    Process entire dataset and compute surprisals for each premise-hypothesis pair.

    For each sample in the dataset, this function:
    1. Tokenizes the hypothesis into words
    2. Computes word-level surprisals with/without premise context
    3. Computes next-word prediction surprisals with/without premise context
    4. Stores all results in a structured format

    Args:
        input_file (str): Path to input TSV file with columns: premise, hypothesis, label
        output_file (str): Path to save compressed JSON results
        model: Loaded language model
        tokenizer: Model tokenizer
        detokenizer: NLTK detokenizer
        config: Configuration object
        logger: Logger for tracking progress
    """
    logger.info(f"Processing {input_file}...")

    # Load TSV data
    df = pd.read_csv(input_file, sep='\t')
    results = {}

    # Process each premise-hypothesis pair
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing surprisals"):
        premise = row['premise']
        hypothesis = row['hypothesis']
        label = row['label']

        # Compute word-level surprisals (actual words in hypothesis)
        surprisals_with_p, surprisals_without_p = compute_surprisals_given_words(
            premise, hypothesis, model, tokenizer, detokenizer, config
        )

        # # Compute next-word prediction surprisals (most likely words at each position)
        next_word_results = compute_surprisals_next_words(
            premise, hypothesis, model, tokenizer, detokenizer, config
        )

        # Store comprehensive results for this sample
        result = {
            'premise': premise,
            'hypothesis': hypothesis,
            'surprisals_with_premise': surprisals_with_p,
            'surprisals_without_premise': surprisals_without_p,
            **next_word_results
        }

        # Add label if available
        if label is not None:
            result['label'] = label

        results[idx] = result

    # Save results as compressed JSON
    logger.info(f"Saving results to {output_file}...")
    helper.save_json_gzip(results, output_file)

    logger.info(f"Processed {len(results)} samples.")


def main():
    """
    Main function to run the LLM surprisal computations.

    This function:
    1. Initializes configuration and sets random seeds for reproducibility
    2. Sets up logging and I/O paths
    3. Loads the specified language model and tokenizer
    4. Configures BOW token detection for word boundary identification
    5. Processes the dataset and saves results
    """

    # ===============================
    # Configuration and Setup
    # ===============================

    # Load configuration from command line arguments
    config = Config()
    config.device = get_device()

    # Set random seed for reproducible results across runs
    set_random_seed(config.seed)

    # Define input/output file paths
    input_file = os.path.join(config.DATA_PATH, 'snli', 'snli_1000_con_test.tsv')
    output_file = os.path.join(config.RESULTS_PATH, config.llm_name, 'snli_1000_con_test.json.gz')
    log_file = os.path.join(config.LOG_PATH, config.llm_name, 'snli_1000_con_test.log')

    # Initialize logging
    logger = helper.init_logger(log_file)
    for k, v in config.__dict__.items():
        logger.info(f'{k}: {v}')

    # ===============================
    # Model Setup
    # ===============================

    model_path = config.LLM_PATH
    print(f"Loading model from: {model_path}")
    print(f"Using device: {config.device}")

    # Load model and tokenizer with appropriate BOW prefix configuration
    model, tokenizer, bow_prefix, bow_prefix_id = setup_model_and_tokenizer(
        model_path, config.llm, config.device
    )

    # Configure model-specific tokens for word boundary detection
    config.bow_prefix_id = bow_prefix_id
    config.bow_token_ids = get_bow_token_ids(bow_prefix, bow_prefix_id, tokenizer)
    config.mid_token_ids = get_mid_token_ids(bow_prefix, tokenizer)

    print(f"BOW prefix: '{bow_prefix}', BOW tokens: {len(config.bow_token_ids)}, MID tokens: {len(config.mid_token_ids)}")

    # ===============================
    # Data Processing
    # ===============================

    print(f"Processing {input_file} -> {output_file}")

    # Initialize detokenizer for text reconstruction
    detokenizer = TreebankWordDetokenizer()

    # Process the entire dataset
    process_dataset(
        input_file, output_file, model, tokenizer, detokenizer, config, logger
    )

    logger.info("Done!")


if __name__ == '__main__':
    fire.Fire(main)