#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Ning Shi'
__email__ = 'mrshininnnnn@gmail.com'

# dependency
# built-in
import math, string
# public
import torch
import torch.nn.functional as F
from nltk.tokenize import word_tokenize

# The Beam class implements a beam search algorithm for generating sequences of tokens.
class Beam:
    def __init__(self, token_ids, token_probs, input_ids, config, parent=None):
        """
        Args:
            token_ids (List[int]): List of token IDs generated so far
            token_probs (List[float]): List of probabilities for each token in the beam
            input_ids (List[int]): Full input IDs (prompt + generated tokens)
            parent (Optional[Beam]): Parent beam for backtracking
        """
        self.token_ids = token_ids
        self.token_probs = token_probs
        self.token_log_probs = [math.log(p) if p > 0 else float('-inf') for p in token_probs]
        self.input_ids = input_ids
        self.parent = parent
        self.bow_prefix_id = config.bow_prefix_id
        self.config = config

    def extend(self, next_token_id, next_token_prob):
        """Return a new Beam with one more token added."""
        return Beam(
            self.token_ids + [next_token_id],
            self.token_probs + [next_token_prob],
            self.input_ids + [next_token_id], 
            self.config, 
            parent=self
        )

    def prob(self):
        """Return product of token probabilities (pseudo-probability)."""
        return math.prod(self.token_probs) if self.token_ids else .0

    def log_prob(self):
        """Return sum of log probabilities (more stable for ranking)."""
        return sum(self.token_log_probs)

    @property
    def done(self):
        """A beam is done if the last token is a BOW token (end of word)."""
        return self.token_ids and self.token_ids[-1] == self.bow_prefix_id

    def path(self):
        """Return a list of beam nodes from root to this beam."""
        beam, result = self, []
        while beam:
            result.append(beam)
            beam = beam.parent
        return list(reversed(result))

    def decoded(self, tokenizer):
        """Decode the beam's token sequence using a tokenizer."""
        return tokenizer.decode(self.token_ids)

    def tokens(self, tokenizer):
        """Return a list of token strings."""
        return [tokenizer.decode([t]) for t in self.token_ids]

    def __eq__(self, other):
        return isinstance(other, Beam) and self.token_ids == other.token_ids

    def __hash__(self):
        return hash(tuple(self.token_ids))

    def __repr__(self):
        return f"Beam(tokens={self.token_ids}, prob={self.prob():.8f}, log_prob={self.log_prob():.8f})"

# The lm function computes the next-token probability distribution for a given input.
def lm(text_or_ids, model, tokenizer, config):
    """
    Compute the next-token probability distribution for a given input.

    Args:
        text_or_ids (str or List[int]): Input string or list of token IDs
        model: Hugging Face AutoModelForCausalLM
        tokenizer: Corresponding tokenizer
        config: Should contain `.device`

    Returns:
        probs (Tensor): Softmax probability distribution over vocabulary, shape [vocab_size]
    """
    # Decode token IDs to text if input is not a string
    if not isinstance(text_or_ids, str):
        text = tokenizer.decode(text_or_ids)
    else:
        text = text_or_ids

    # Tokenize input
    x = tokenizer(text, return_tensors="pt")
    input_ids = x["input_ids"].to(config.device)
    attention_mask = x["attention_mask"].to(config.device)

    # Run model and get logits for the last token
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits[:, -1, :]  # Last token position
    probs = F.softmax(logits, dim=-1).squeeze()  # [vocab_size]

    return probs

# The norm_probs function normalizes a probability distribution over a set of valid token IDs.
def norm_probs(probs, valid_token_ids):
    """
    Normalize probs over a precomputed set of valid token IDs.

    Args:
        probs (Tensor): Raw probability distribution over vocabulary, shape [vocab_size]
        valid_token_ids (List[int]): Token IDs that are considered valid for normalization

    Returns:
        norm_probs (Tensor): New probability distribution normalized over valid_token_ids
    """
    masked_probs = torch.zeros_like(probs)
    masked_probs[valid_token_ids] = probs[valid_token_ids]
    total = masked_probs.sum()

    return masked_probs / total

def inject_eow_prob(probs, bow_token_ids, bow_prefix_id):
    """
    Inject EOW probability into the bow_prefix_id slot by reallocating 
    the total BOW mass there, and zeroing out the original BOW tokens.
    
    Returns a new probability tensor (not in-place).
    """
    probs = probs.clone()
    bow_mass = probs[bow_token_ids].sum()
    probs[bow_token_ids] = 0.0
    probs[bow_prefix_id] = bow_mass
    return probs

def is_bow_token(token_id, bow_prefix, tokenizer) -> bool:
    """
    Returns True if the token is a valid token:
    - start with the BOW prefix
    - contains only alphanumeric characters
    """
    return tokenizer.convert_ids_to_tokens(token_id).startswith(bow_prefix)

def get_bow_token_ids(bow_prefix, bow_prefix_id, tokenizer) -> list:
    """
    Returns a list of token IDs that:
    - start with the BOW prefix (e.g., Ġ), AND
    - decode to alphabetic strings (i.e., isalpha())
    """
    vocab_size = tokenizer.vocab_size
    bow_token_ids = [i for i in range(vocab_size) if is_bow_token(i, bow_prefix, tokenizer)]
    bow_token_ids.remove(bow_prefix_id)
    return bow_token_ids

def is_mid_token(token_id, bow_prefix, tokenizer, pun=False, non_ascii=False) -> bool:
    """
    Returns True if the token is a valid continuation of a word:
    - does NOT start with the BOW prefix
    - contains only alphabetic characters

    Args:
        token_id: Token ID to check
        bow_prefix: Beginning-of-word prefix (e.g., 'Ġ')
        tokenizer: Tokenizer instance
        pun: If True, allow punctuation (for words like "N'Djamena", "Sana'a")
        non_ascii: If True, allow non-ASCII alphabetic chars (é, í, ñ, etc.)
                         If False, restrict to ASCII-only (a-z, A-Z)
    """
    token = tokenizer.convert_ids_to_tokens(token_id)
    token_str = tokenizer.decode(token_id)
    # take care of some special cases
    if pun:
        # N'Djamena; Sana'a
        token_str = ''.join([l for l in token_str if l not in string.punctuation])
        # allow punctuation in the token string
        if token_str in string.punctuation:
            return True
    # does NOT start with the BOW prefix
    if token.startswith(bow_prefix):
        return False
    # Check if the token is alphabetic
    if not token_str.isalpha():
        return False
    # Check ASCII constraint if required
    if not non_ascii:
        if not all(ord(c) < 128 for c in token_str):
            return False
    # If it passes all checks, it's a valid mid-token
    return True


def get_mid_token_ids(bow_prefix, tokenizer, pun=False, non_ascii=False) -> list:
    """
    Returns a list of token IDs that:
    - do NOT start with the BOW prefix (e.g., Ġ)
    - decode to alphabetic strings (i.e., isalpha())

    Args:
        bow_prefix: Beginning-of-word prefix (e.g., 'Ġ')
        tokenizer: Tokenizer instance
        pun: If True, allow punctuation in tokens
        non_ascii: If True, allow non-ASCII alphabetic chars (é, í, ñ, etc.)
    """
    vocab_size = tokenizer.vocab_size
    return [i for i in range(vocab_size) if is_mid_token(i, bow_prefix, tokenizer, pun, non_ascii)]

# The get_sentence_beams function generates beams for each word in a sentence.
def get_sentence_beams(text, tokenizer, detokenizer, model, config, context=''):
    ws = word_tokenize(text)
    h_beams = []
    for i in range(len(ws)):
        prev_ws = ws[:i]
        tgt_w = ws[i]
        tgt_tks = tokenizer(' ' + tgt_w, add_special_tokens=False).input_ids + [config.bow_prefix_id]
        full_context = context + ' ' + detokenizer.detokenize(prev_ws) if i else context

        # Step 1: Initialize input and beam
        input_ids = tokenizer(full_context, return_tensors="pt").input_ids.tolist()[0]
        input_ids = [config.bow_prefix_id] if not input_ids else input_ids

        beam = Beam([], [], input_ids, config)
        # Step 2: Beam search decoding
        for depth, tgt_token in enumerate(tgt_tks):
            # Step 2.1: Get next-token probability distribution
            next_probs = lm(beam.input_ids, model, tokenizer, config)
            # Step 2.2: Inject end-of-word (EOW) probability at depths > 0
            if depth:
                next_probs = inject_eow_prob(next_probs, config.bow_token_ids, config.bow_prefix_id)
            # Step 2.3: Normalize probabilities over valid token IDs
            vocab_ids = config.bow_token_ids if depth == 0 else config.mid_token_ids + [config.bow_prefix_id]
            next_probs = norm_probs(next_probs, vocab_ids)
            # Step 2.4: get the target token probabilities
            tgt_prob = next_probs[tgt_token].item()
            beam = beam.extend(tgt_token, tgt_prob)

        h_beams.append(beam)

    return h_beams

# The get_tgt_words_beams function generates beams for a list of target words given a context.
def get_tgt_words_beams(context, words, tokenizer, model, config, sort=False):
    words = [' ' + w for w in words]
    tk_words = [tokenizer(w, return_tensors="pt", add_special_tokens=False).input_ids.tolist()[0] for w in words]
    # padding
    beam_depth = max([len(t) for t in tk_words]) + 1  # +1 for the EOS token
    for i in range(len(tk_words)):
        tk_words[i] += [config.bow_prefix_id] * (beam_depth - len(tk_words[i]) + 1)
    # Step 1: Initialize input and beam
    input_ids = tokenizer(context, return_tensors="pt").input_ids.tolist()[0]

    beams = [Beam([], [], input_ids, config)] * len(words)
    # Step 2: Beam search decoding
    for depth in range(beam_depth):
        for width, beam in enumerate(beams):
            if beam.done:
                continue
            # Step 2.1: Get next-token probability distribution
            next_probs = lm(beam.input_ids, model, tokenizer, config)
            # Step 2.2: Inject end-of-word (EOW) probability at depths > 0
            if depth:
                next_probs = inject_eow_prob(next_probs, config.bow_token_ids, config.bow_prefix_id)
            # Step 2.3: Normalize probabilities over valid token IDs
            vocab_ids = config.bow_token_ids if depth == 0 else config.mid_token_ids + [config.bow_prefix_id]
            next_probs = norm_probs(next_probs, vocab_ids)
            # Step 2.4: get the target token probabilities
            t = tk_words[width][depth]
            t_prob = next_probs[t].item()
            beams[width] = beam.extend(t, t_prob)
        if all(b.done for b in beams):
            break
    if sort:
        beams = sorted(beams, key=lambda beam: -beam.prob())
    return beams

def get_topk_beams(context, tokenizer, model, config, beam_width=100, max_depth=10):
    """
    Get the top-k most likely next word beams given a context using beam search.
    Args:
        context (str): The input context
        tokenizer: Hugging Face tokenizer
        model: Hugging Face model
        config: Configuration object with bow_token_ids, etc.
        beam_width (int): Number of beams to keep at each step
        max_depth (int): Maximum depth for beam search  
    Returns:
        List[Beam]: List of beams for the most probable next words, sorted by probability
    """
    # Step 1: Initialize input and beam
    input_ids = tokenizer(context, return_tensors="pt").input_ids.tolist()[0]
    input_ids = [config.bow_prefix_id] if not input_ids else input_ids

    beams = [Beam([], [], input_ids, config)]

    # Step 2: Beam search decoding
    for depth in range(max_depth):
        new_beams = []
        for beam in beams:
            if beam.done:
                new_beams.append(beam)
                continue
            # Step 2.1: Get next-token probability distribution
            next_probs = lm(beam.input_ids, model, tokenizer, config)
            # Step 2.2: Inject end-of-word (EOW) probability at depths > 0
            if depth:
                next_probs = inject_eow_prob(next_probs, config.bow_token_ids, config.bow_prefix_id)
            # Step 2.3: Normalize probabilities over valid token IDs
            vocab_ids = config.bow_token_ids if depth == 0 else config.mid_token_ids + [config.bow_prefix_id]
            next_probs = norm_probs(next_probs, vocab_ids)
            # Step 2.4: Top-k expansion
            topk_probs, topk_ids = torch.topk(next_probs, k=beam_width)
            for topk_id, topk_prob in zip(topk_ids.tolist(), topk_probs.tolist()):
                new_beams.append(beam.extend(topk_id, topk_prob))
        # Step 3: Keep top-scoring beams
        beams = sorted(new_beams, key=lambda beam:-beam.prob())[:beam_width]
        if all(b.done for b in beams):
            break

    # Filter only completed beams (words)
    completed_beams = [beam for beam in beams if beam.done]
    return sorted(completed_beams, key=lambda beam: -beam.prob())

def get_sentence_topk_beams(text, tokenizer, detokenizer, model, config, context=''):
    """
    Get the most likely next word beam at each position in a sentence.
    Similar to get_sentence_beams but returns the most probable next word at each position.

    Args:
        text (str): The target sentence
        tokenizer: Hugging Face tokenizer
        detokenizer: NLTK detokenizer
        model: Hugging Face model
        config: Configuration object
        context (str): Context to prepend before the sentence

    Returns:
        List[Beam]: List of beams for the most probable next words at each position
    """
    ws = word_tokenize(text)
    next_beams = []

    for i in range(len(ws)):
        prev_ws = ws[:i]
        full_context = context + ' ' + detokenizer.detokenize(prev_ws) if i else context
        full_context = full_context.strip()

        # Get the most likely next word at this position
        top_beams = get_topk_beams(full_context, tokenizer, model, config, beam_width=10, max_depth=5)
        if top_beams:
            next_beams.append(top_beams[0])
        else:
            # Create empty beam if no next word found
            input_ids = tokenizer(full_context, return_tensors="pt").input_ids.tolist()[0]
            input_ids = [config.bow_prefix_id] if not input_ids else input_ids
            empty_beam = Beam([], [], input_ids, config)
            next_beams.append(empty_beam)

    return next_beams