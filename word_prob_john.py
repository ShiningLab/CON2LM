import re
import csv
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "deepseek-ai/DeepSeek-V2-Lite"
# MODEL_NAME = "mistralai/Mistral-7B-v0.1"
# MODEL_NAME = "meta-llama/Meta-Llama-2-7B-hf"
TOP_K = 10
FIRST_N_TOKENS = 3

INPUT_FILE = "./data/wikidata/capitals-mini.csv"
OUTPUT_FILE = "./data/nextwords/capitals-mini.csv"

# Llama Models
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=True)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     torch_dtype=torch.float16,
#     device_map="auto",
#     attn_implementation="sdpa",
#     trust_remote_code=True,
#     use_auth_token=True,
# )

# DeepSeek Models
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, low_cpu_mem_usage=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
)


def is_valid_token(token):
    return bool(re.match(r"^[a-zA-ZĠ\s]+(?:[a-zA-Z0-9Ġ\s]*)$", token))


def clean_token(token_str):
    return " ".join(token_str.replace("Ġ", " ").split())


def get_next_word_probabilities(input_text, top_k=TOP_K):
    """
    Get next word predictions for input text:
    1. tokenize input text
    2. initialize input_ids and attention_mask for top_k predictions
    3. generate first token for top_k predictions to get top_k different inputs
    4. loop through each input and generate next tokens up to FIRST_N_TOKENS to ensure a full word is generated
    5. use the Ġ character or space to determine end of the first word
    6. compute the conditional word probabilities of each using the token probabilities
    7. sort the words by probability
    8. return the top_k words

    Args:
        input_text (str): The input text to get next word predictions for
        top_k (int): Number of top predictions to return

    Returns:
        list: List of tuples (word, probability) sorted by probability
    """

    # Tokenize input text
    input = tokenizer(input_text, return_tensors="pt")
    input_ids = input["input_ids"].to(model.device)
    attention_mask = input["attention_mask"].to(model.device)

    # Initialize arrays for top_k predictions
    input_list = []
    attention_mask_list = []
    token_results = [[] for _ in range(FIRST_N_TOKENS)]
    word_results = []

    # ======= Generate first token predictions to initialize first k inputs =======
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    logits = outputs.logits[:, -1, :]
    probs = F.softmax(logits, dim=-1)
    topk_probs, topk_tokens = torch.topk(probs, top_k, dim=-1)

    # Store first token results and prepare inputs for next iteration
    for token_id, prob in zip(topk_tokens[0], topk_probs[0]):
        token_str = tokenizer.decode([token_id])
        if is_valid_token(token_str):
            token_results[0].append(([token_str], prob.item()))
            new_input_ids = torch.cat([input_ids, token_id.view(1, 1)], dim=1)
            new_attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), device=model.device)], dim=1
            )
            input_list.append(new_input_ids)
            attention_mask_list.append(new_attention_mask)

    # Debug
    print("\nSelected tokens for position 0:")
    for sequence, prob in token_results[0]:
        print(f"Sequence: {' '.join(sequence)}, Probability: {prob:.8f}")

    # Track completed sequences; completed[i] = True if input_list[i] has been completed; completed if input_list[i] has generated first word
    completed = [False] * len(input_list)

    # ======= Generate next tokens for each input iteration =======
    for token_idx in range(1, FIRST_N_TOKENS):
        current_tokens = []
        new_input_list = []
        new_attention_mask_list = []
        new_completed = []

        # Debug
        print(f"\n=== Position {token_idx} Tokens ===")

        # Process each input sequence and calculate conditionals using previous probabilities
        for i, (prev_sequence, prev_prob) in enumerate(token_results[token_idx - 1]):
            if completed[i]:
                continue

            with torch.no_grad():
                outputs = model(input_list[i], attention_mask=attention_mask_list[i])

            logits = outputs.logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            # Get next token and check if it starts a new word
            max_prob, next_token_id = torch.max(probs, dim=-1)
            next_token_id = next_token_id.item()
            next_token_str = tokenizer.decode([next_token_id])

            # If token starts with space/Ġ, add current word to results
            if next_token_str.startswith(" ") or next_token_str.startswith("Ġ"):
                word = clean_token("".join(prev_sequence))
                word_results.append((word, prev_prob))
                continue

            # Update sequence and prepare for next iteration
            new_sequence = prev_sequence + [next_token_str]
            conditional_prob = max_prob.item() * prev_prob
            current_tokens.append((new_sequence, conditional_prob))

            new_token = torch.tensor([[next_token_id]], device=model.device)
            new_input_ids = torch.cat([input_list[i], new_token], dim=1)
            new_attention_mask = torch.cat(
                [attention_mask_list[i], torch.ones((1, 1), device=model.device)], dim=1
            )

            new_input_list.append(new_input_ids)
            new_attention_mask_list.append(new_attention_mask)
            new_completed.append(False)

        # Update lists for next iteration
        input_list = new_input_list
        attention_mask_list = new_attention_mask_list
        completed = new_completed
        token_results[token_idx] = current_tokens

        # Debug
        print(f"\nSelected tokens for position {token_idx}:")
        for sequence, prob in current_tokens:
            print(f"Sequence: {' '.join(sequence)}, Probability: {prob:.8f}")

    # Add any remaining sequences to results
    for sequence, prob in token_results[-1]:
        word = clean_token("".join(sequence))
        if word:
            word_results.append((word, prob))

    # Return top-k words sorted by probability
    word_results.sort(key=lambda x: x[1], reverse=True)
    return word_results[:top_k]


# ======= Process prompts =======
if __name__ == "__main__":
    with open(INPUT_FILE, "r") as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)

        with open(OUTPUT_FILE, "w") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(["source", "top_k_predictions"])
            for row in tqdm(rows, desc="Processing prompts", unit="prompt"):
                input_text = row["source"]
                word_probs = get_next_word_probabilities(input_text)
                top_k_words = "; ".join(
                    [f"{word} ({prob:.8f})" for word, prob in word_probs]
                )
                writer.writerow([input_text, top_k_words])