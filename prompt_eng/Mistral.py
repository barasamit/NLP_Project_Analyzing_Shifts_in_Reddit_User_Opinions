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
import json


def get_next_filename(base_name, extension):
    i = 1
    while os.path.exists(f"{base_name}_{i}.{extension}"):
        i += 1
    return f"{base_name}_{i}.{extension}"


def evaluate_performance(classified_csv, actual_csv):
    # Load both CSV files
    classified_df = pd.read_csv(classified_csv)  # Handle missing values

    actual_df = pd.read_csv(actual_csv)

    if 'classification' not in classified_df.columns or 'Label' not in actual_df.columns:
        raise ValueError(
            "Classified CSV must have a 'classification' column and Actual CSV must have a 'Label' column.")

    # Ensure alignment of data
    if len(classified_df) != len(actual_df):
        if len(classified_df) > len(actual_df):
            classified_df = classified_df.iloc[:len(actual_df)]
        else:
            actual_df = actual_df.iloc[:len(classified_df)]
        # raise ValueError("The number of rows in the classified and actual CSV files must match.")

    # Calculate performance metrics
    y_pred = classified_df['classification'].astype(int)
    y_true = actual_df['Label'].astype(int)

    # Define the label order as requested: 0, 1, -1
    label_order = [0, 1, -1]
    label_names = ["Neutral (0)", "Pro-Israel (1)", "Pro-Palestine (-1)"]

    # Generate classification report
    report = classification_report(y_true, y_pred,
                                   labels=label_order,
                                   target_names=label_names)
    print("\nPerformance Evaluation:")
    print(report)

    # Generate confusion matrix with specified label order
    cm = confusion_matrix(y_true, y_pred, labels=label_order)

    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot the normalized confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=label_names,
                yticklabels=label_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()  # Ensure labels are not cut off
    plt.savefig("confusion_matrix_normalized.png", bbox_inches='tight')
    plt.show()


def classify_comments_with_gpt(input_csv, base_output_name, prompt_file, batch_size=20):
    mistral_models_path = Path.home().joinpath('mistral_models', '7B-Instruct-v0.3')
    mistral_models_path.mkdir(parents=True, exist_ok=True)
    # llama_models_path = Path.home().joinpath('llama_models', 'llama-7b')
    # llama_models_path.mkdir(parents=True, exist_ok=True)

    snapshot_download(repo_id="mistralai/Mistral-7B-Instruct-v0.3",
                      allow_patterns=["params.json", "consolidated.safetensors", "tokenizer.model.v3"],
                      local_dir=mistral_models_path)

    tokenizer = MistralTokenizer.from_file(f"{mistral_models_path}/tokenizer.model.v3")
    model = Transformer.from_folder(mistral_models_path)

    df = pd.read_csv(input_csv)
    output_csv = get_next_filename(base_output_name, "csv")

    if 'self_text' not in df.columns:
        raise ValueError("The CSV must have a column named 'self_text' containing the Reddit comments.")

    with open(prompt_file, 'r', encoding="utf-8") as file:
        base_prompt = file.read()

    classifications = []
    flag = 0
    for i in range(0, len(df), batch_size):
        end_idx = min(i + batch_size, len(df))
        batch = df['self_text'][i:end_idx].tolist()
        formatted_comments = "\n".join(f"{idx + 1}: {comment}" for idx, comment in enumerate(batch))

        full_prompt = f"{base_prompt}\n\nHere are the comments:\n{formatted_comments}\n\nProvide your response as a JSON list of integers corresponding to the classification of each comment. DO NOT include any other information in your response!!!!!"
        if flag == 0:
            print(full_prompt)
            flag = 1
        try:

            completion_request = ChatCompletionRequest(
                messages=[UserMessage(content=full_prompt)])

            tokens = tokenizer.encode_chat_completion(completion_request).tokens

            out_tokens, _ = generate([tokens], model, max_tokens=64, temperature=0.0,
                                     eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
            result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
            result = ast.literal_eval(result)
            print(f"Classified batch starting at index {i}")
            print(result)
            print(type(result))

            classifications.extend(result)

        except Exception as e:
            print(f"Error classifying batch starting at index {i}. Error: {e}")
            classifications.extend([0] * len(batch))  # Assign default classification (e.g., Neutral: 0)

        print(classifications)

    if len(classifications) != len(df):
        print(
            f"Warning: Number of classifications ({len(classifications)}) does not match number of input rows ({len(df)}).")
        missing_count = len(df) - len(classifications)
        classifications.extend([0] * missing_count)  # Fill missing with default classification

    result_df = pd.DataFrame({'classification': classifications})
    result_df.to_csv(output_csv, index=False)

    print(f"Classifications saved to {output_csv}")
    return output_csv

    result_df = pd.DataFrame({'classification': classifications})
    result_df.to_csv(output_csv, index=False)

    print(f"Classifications saved to {output_csv}")
    return output_csv


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


def process_file(input_file, prompt, tokenizer, model,
                 output_dir=r"C:\Users\oren fix\OneDrive\Desktop\study\second_year_MSc\NLP\project"):
    with open(input_file, 'r') as f:
        data = json.load(f)

    for post_id, post_data in data.items():
        if "submission" in post_data:
            submission = post_data["submission"]
            content = f"{submission.get('title', '')} {submission.get('selftext', '')}"
            submission["stance"] = classify_single_sentence_with_gpt(content, prompt, tokenizer, model)

        if "comments" in post_data:
            for comment_id, comments in post_data["comments"].items():
                for comment in comments:
                    content = comment.get("body", "")
                    comment["stance"] = classify_single_sentence_with_gpt(content, prompt,tokenizer,model)

    output_filename = os.path.join(output_dir, f"labeled_{os.path.basename(input_file)}")
    os.makedirs(output_dir, exist_ok=True)

    # with open(output_filename, 'w') as f:
    #     json.dump(data, f, indent=2)

    return output_filename


output_csv = classify_comments_with_gpt("test_data_new.csv", "test_classifications_zero_shot", "cot.txt", batch_size=10)

evaluate_performance(output_csv, "test_data_new.csv")

# if __name__ == "__main__":
#     # Load the tokenizer and model
#     tokenizer, model = load_mistral_model()
#
#     # Path to your uploaded file
#     file_path = r"cleaned_IsraelPalestine_2023-09_full_data.json"
#
#     # # Open and read the JSON file
#     # try:
#     #     with open(file_path, 'r', encoding='utf-8') as file:
#     #         data = json.load(file)
#     #         print("File loaded successfully!")
#     #         # Display a small part of the data for verification
#     #         print(json.dumps(data, indent=4)[:500])  # Display first 500 characters
#     # except Exception as e:
#     #     print(f"An error occurred: {e}")
#
#     process_file(file_path, "prompt_few_shot.txt", tokenizer, model)
#
#     # # Classify a single sentence
#     # example_sentence = "I love you!"
#     # prompt_file = "prompt_few_shot.txt"
#     # label = classify_single_sentence_with_gpt(
#     #     sentence=example_sentence,
#     #     prompt_file=prompt_file,
#     #     tokenizer=tokenizer,
#     #     model=model
#     # )
#     # print(f"Classification for sentence '{example_sentence}': {label}")
