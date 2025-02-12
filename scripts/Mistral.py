from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from transformers import AutoTokenizer, AutoModelForCausalLM
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
import pandas as pd
from huggingface_hub import snapshot_download
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import ast
from tqdm import tqdm

def load_mistral_model(model_path: Path = None):
    """
    Loads Mistral model & tokenizer. Returns (tokenizer, model).
    By default, downloads to ~/mistral_models/7B-Instruct-v0.3 if not found.
    """
    if model_path is None:
        model_path = Path.home().joinpath('mistral_models', '7B-Instruct-v0.3')
    model_path.mkdir(parents=True, exist_ok=True)

    # Download model files if not already present
    snapshot_download(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        allow_patterns=[
            "params.json",
            "consolidated.safetensors",
            "tokenizer.model.v3"
        ],
        local_dir=model_path
    )

    # Load tokenizer and model
    tokenizer = MistralTokenizer.from_file(str(model_path / "tokenizer.model.v3"))
    model = Transformer.from_folder(model_path)

    return tokenizer, model

def classify_single_sentence_with_gpt(sentence: str, prompt_file: str, tokenizer, model) -> int:
    """
    Classify a single sentence into 1, 0, or -1 using Mistral.
      - sentence: the text to classify.
      - prompt_file: text file with base few-shot or zero-shot instructions.
      - tokenizer, model: from load_mistral_model() to avoid re-loading each time.

    Returns:
      An integer label (1, 0, or -1).
    """

    # Load the prompt instructions (few-shot or zero-shot) from a file
    with open(prompt_file, 'r', encoding="utf-8") as f:
        base_prompt = f.read().strip()

    # Construct the final prompt for a single sentence
    # We ask for a JSON list of length 1, or a single integer in JSON, depending on your style.
    # Here we'll request a single list with one integer for consistency with your previous code.
    full_prompt = (
        f"{base_prompt}\n\n"
        f"Here is the sentence:\n1: {sentence}\n\n"
        f"Provide your response as a JSON list of ONE integer corresponding "
        f"to the classification of the above sentence. "
        f"DO NOT include any other information in your response."
    )

    # Prepare the request for Mistral
    completion_request = ChatCompletionRequest(
        messages=[UserMessage(content=full_prompt)]
    )

    # Encode the prompt
    input_tokens = tokenizer.encode_chat_completion(completion_request).tokens

    # Generate the output
    out_tokens, _ = generate(
        [input_tokens],
        model,
        max_tokens=32,
        temperature=0.0,
        eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id
    )

    # Decode
    raw_output = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])

    # Safely parse JSON
    try:
        # Expect something like: [1]
        parsed = ast.literal_eval(raw_output)
        if isinstance(parsed, list) and len(parsed) == 1:
            # Return the first integer
            return int(parsed[0])
        else:
            # If parsing didn't match our expected structure,
            # fallback to neutral
            return 0
    except Exception as e:
        print("Parsing error:", e)
        return 0  # Fallback to 0 (neutral)
    
def process_file(input_file, prompt, tokenizer, model,
                 output_dir=r"/home/meirio/nlp/cleaned_and_labeled_reddit_files"):
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Add progress bar for submissions
    submission_count = sum(1 for post in data.values() if "submission" in post)
    submission_pbar = tqdm(total=submission_count, desc="Processing submissions")

    # Count total comments for progress bar
    comment_count = sum(
        len(comments) 
        for post in data.values() 
        for comment_list in post.get("comments", {}).values() 
        for comments in comment_list
    )
    comment_pbar = tqdm(total=comment_count, desc="Processing comments")

    for post_id, post_data in data.items():
        if "submission" in post_data:
            submission = post_data["submission"]
            content = f"{submission.get('title', '')} {submission.get('selftext', '')}"
            submission["stance"] = classify_single_sentence_with_gpt(content, prompt, tokenizer, model)
            submission_pbar.update(1)

        if "comments" in post_data:
            for comment_id, comments in post_data["comments"].items():
                for comment in comments:
                    content = comment.get("body", "")
                    comment["stance"] = classify_single_sentence_with_gpt(content, prompt, tokenizer, model)
                    comment_pbar.update(1)

    submission_pbar.close()
    comment_pbar.close()

    output_filename = os.path.join(output_dir, f"labeled_{os.path.basename(input_file)}")
    os.makedirs(output_dir, exist_ok=True)

    with open(output_filename, 'w') as f:
        json.dump(data, f, indent=2)

    return output_filename

# Load the tokenizer and model
tokenizer, model = load_mistral_model()
file_path = "/home/meirio/nlp/cleaned_reddit_files/cleaned_IsraelPalestine_2024-02_full_data.json"
out_file_name = process_file(file_path, "/home/meirio/nlp/prompt_few_shot.txt", tokenizer, model)
print(f"Output file saved to: {out_file_name}")