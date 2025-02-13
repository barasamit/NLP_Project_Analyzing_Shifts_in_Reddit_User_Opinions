To run this function, you need to execute it from the main script.

Below is an example of how to classify a single sentence using the function, which takes an input sentence and returns a classification:

    # # Classify a single sentence
example_sentence = "I love you!"
prompt_file = "prompt_few_shot.txt"
label = classify_single_sentence_with_gpt(
sentence=example_sentence,
prompt_file=prompt_file,
tokenizer=tokenizer,
model=model
)
print(f"Classification for sentence '{example_sentence}': {label}")
