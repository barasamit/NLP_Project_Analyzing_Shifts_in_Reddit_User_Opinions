# Mistral-Based Text Classification

## Overview
This project uses the Mistral 7B-Instruct model to classify text into predefined categories. It supports batch classification of comments from CSV files and single-sentence classification. The implementation includes model loading, inference, and performance evaluation with classification reports and confusion matrices.

## Features
- Classifies text as **Neutral (0), Pro-Israel (1), or Pro-Palestine (-1)**.
- Supports **batch classification** from CSV files.
- **Single-sentence classification** using a few-shot or zero-shot prompt.
- Evaluates model performance with **confusion matrices** and **classification reports**.

## Installation
Ensure you have Python 3.8+ installed and then install dependencies:
```sh
pip install transformers pandas scikit-learn seaborn matplotlib huggingface_hub mistral_inference
```

## Usage
### 1. Classify Comments from CSV
```python
output_csv = classify_comments_with_gpt("input_data.csv", "classified_output", "prompt.txt", batch_size=10)
```

### 2. Evaluate Performance
```python
evaluate_performance("classified_output.csv", "actual_labels.csv")
```

### 3. Classify a Single Sentence
```python
label = classify_single_sentence_with_gpt("I love this place!", "prompt.txt", tokenizer, model)
print(f"Classification: {label}")
```

## Model Setup
The Mistral model is automatically downloaded if not found locally. To manually set up the model:
```python
tokenizer, model = load_mistral_model()
```

## Notes
- Ensure the input CSV file contains a column named `self_text` for batch classification.
- The classification output is saved as a CSV file with a `classification` column.

## License
MIT License


