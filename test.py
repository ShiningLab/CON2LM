#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Test script for the CON2LM token-to-word decoding algorithm.

This script demonstrates:
1. How subword tokenization splits words into multiple tokens
2. How the token-to-word algorithm computes word-level probabilities
3. Word surprisal computation for contradiction detection

Usage:
    python test.py
"""

__author__ = 'Ning Shi'
__email__ = 'mrshininnnnn@gmail.com'

# Built-in imports
import os
import sys

# Public library imports
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.tokenize.treebank import TreebankWordDetokenizer

# Private module imports
from config import Config
from src.helper import get_device, set_random_seed
from src.model_configs import BOW_PREFIX_MAP, DEFAULT_BOW_PREFIX
from src.con2lm import (
    get_bow_token_ids,
    get_mid_token_ids,
    get_sentence_beams,
    get_topk_beams
)


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demonstrate_subword_tokenization(tokenizer, model_name):
    """Show how words are split into subword tokens."""
    print_section("1. Subword Tokenization Problem")

    print(f"\nModel: {model_name} | BOW prefix: '{BOW_PREFIX_MAP.get(model_name, DEFAULT_BOW_PREFIX)}'")

    # Classic examples from the paper

    print("\nPrefix sharing (different meanings):")
    for word in ["mat", "matron"]:
        tokens = tokenizer.tokenize(' ' + word)
        print(f"  '{word}' → {tokens}")

    print("\n→ Problem: Words split into variable numbers of subword tokens")
    print("→ Solution: CON2LM computes P(word) using beam search over tokens")


def demonstrate_word_probability(premise, hypothesis, model, tokenizer, detokenizer, config):
    """Demonstrate word-level probability computation."""
    print_section("2. Computing Word-Level Probabilities")

    context = f"Since {premise[:-1]}, therefore"
    print(f"\nContext: \"{context}\"")
    print(f"Hypothesis: \"{hypothesis}\"")

    # Compute word probabilities
    print("\nComputing word probabilities using beam search...")
    words = hypothesis.split()
    beams = get_sentence_beams(hypothesis, tokenizer, detokenizer, model, config, context=context)

    # Show detailed breakdown for first multi-token word
    multi_token_found = False
    for word, beam in zip(words, beams):
        if len(beam.token_ids) > 1:
            tokens = [tokenizer.decode([tid]) for tid in beam.token_ids[:-1]]
            probs = beam.token_probs[:-1]
            print(f"\nExample: '{word}' → {tokens}")
            for j, (token, prob) in enumerate(zip(tokens, probs), 1):
                print(f"  Token {j}: P('{token}') = {prob:.6f}")
            print(f"  → P(word) = {beam.prob():.8f} (product)")
            print(f"  → Surprisal = {-beam.log_prob():.4f}")
            multi_token_found = True
            break

    if not multi_token_found:
        print("  (All words are single tokens in this example)")

    # Display results table
    print(f"\n{'Word':<20} {'Probability':<15} {'Surprisal':<12}")
    print("-" * 50)
    for word, beam in zip(words, beams):
        print(f"{word:<20} {beam.prob():<15.8f} {-beam.log_prob():<12.4f}")
    print("-" * 50)


def demonstrate_contradiction_detection(model, tokenizer, detokenizer, config):
    """Show how surprisal correlates with contradiction."""
    print_section("3. Contradiction Detection")

    test_cases = [
        ("The cat is sleeping on the couch.", "The cat is resting on the furniture.", "Non-contradiction"),
        ("The cat is sleeping on the couch.", "The cat is running in the garden.", "Contradiction")
    ]

    for i, (premise, hypothesis, label) in enumerate(test_cases, 1):
        print(f"\nExample {i}: {label}")
        print(f"  Premise: \"{premise}\"")
        print(f"  Hypothesis: \"{hypothesis}\"")

        context = f"Since {premise[:-1]}, therefore"
        beams = get_sentence_beams(hypothesis, tokenizer, detokenizer, model, config, context=context)

        words = hypothesis.split()
        surprisals = [-beam.log_prob() for beam in beams]

        print("  Word surprisals:")
        for word, surp in zip(words, surprisals):
            print(f"    {word:<15} {surp:>8.4f}")

        if surprisals:
            print(f"  → Last: {surprisals[-1]:.4f} | Mean: {sum(surprisals)/len(surprisals):.4f} | Max: {max(surprisals):.4f}")

    print("\n→ Key finding: Contradictions show HIGHER surprisal scores")


def demonstrate_top_predictions(context, model, tokenizer, config):
    """Show what the model predicts for next word."""
    print_section("2. Model Predictions")

    print(f"\nGiven: \"{context}\"")
    print("\nTop 5 predicted next words:")

    top_beams = get_topk_beams(context, tokenizer, model, config, beam_width=5, max_depth=10)

    for i, beam in enumerate(top_beams[:5], 1):
        word = beam.decoded(tokenizer).strip()
        surprisal = -beam.log_prob()
        print(f"  {i}. '{word}' (surprisal={surprisal:.4f})")

    print("\n→ Model struggles with negation: predictions may not exclude 'kitchen'")


def main():
    """Run all demonstration examples."""
    print("\n" + "=" * 70)
    print("  CON2LM Token-to-Word Decoding Algorithm Demo")
    print("  Paper: Word Surprisal Correlates with Sentential Contradiction")
    print("=" * 70)

    # Setup
    print("\nInitializing...")
    config = Config()
    config.seed = 0
    config.beam_depth = 10

    # Use a small model for quick testing (Qwen3-4B or whatever is available)
    # Users should modify this path to their local model
    config.llm = 'HF/Qwen3-4B'

    set_random_seed(config.seed)
    config.device = get_device()

    print(f"Device: {config.device}")
    print(f"Model: {config.llm}")

    # Check if model path exists
    if not os.path.exists(config.LLM_PATH):
        print(f"\n⚠ WARNING: Model not found at {config.LLM_PATH}")
        print("\nTo run this test, you need to:")
        print("1. Download a language model (e.g., Qwen3-4B, Llama-3.2-3B)")
        print("2. Place it in res/llms/ directory")
        print("3. Update config.py with the correct path")
        print("\nAlternatively, modify this script to use a Hugging Face model ID:")
        print("  tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen-3-4B')")
        print("  model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen-3-4B')")
        sys.exit(1)

    # Load model
    print(f"\nLoading model from {config.LLM_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(config.LLM_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        config.LLM_PATH,
        device_map=config.device
    )
    model.eval()

    # Setup BOW tokens
    bow_prefix = BOW_PREFIX_MAP.get(config.llm, DEFAULT_BOW_PREFIX)
    config.bow_prefix_id = tokenizer.convert_tokens_to_ids(bow_prefix)
    config.bow_token_ids = get_bow_token_ids(bow_prefix, config.bow_prefix_id, tokenizer)
    config.mid_token_ids = get_mid_token_ids(bow_prefix, tokenizer)

    print(f"BOW prefix: '{bow_prefix}'")
    print(f"BOW tokens: {len(config.bow_token_ids)}")
    print(f"MID tokens: {len(config.mid_token_ids)}")

    detokenizer = TreebankWordDetokenizer()

    # Run demonstrations
    demonstrate_subword_tokenization(tokenizer, config.llm)

    # Use direct example for predictions
    context = "John is not in the kitchen. John is in the"
    demonstrate_top_predictions(context, model, tokenizer, config)

    demonstrate_contradiction_detection(model, tokenizer, detokenizer, config)

    # Final summary
    print_section("Summary")
    print("\n  • Computes word-level probabilities via beam search")
    print("  • Handles subword tokenization with word boundary constraints")
    print("  • Word surprisal correlates with sentential contradiction")
    print("\nPaper: https://aclanthology.org/2026.eacl-long.211.pdf")
    print("Code: src/con2lm.py | Analysis: main.ipynb")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
