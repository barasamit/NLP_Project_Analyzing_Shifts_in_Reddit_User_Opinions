import openai
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def classify_comments_with_gpt(api_key, input_csv, output_csv, batch_size=20):
    # Load the CSV file
    df = pd.read_csv(input_csv)

    if 'self_text' not in df.columns:
        raise ValueError("The CSV must have a column named 'self_text' containing the Reddit comments.")

    # Prepare API
    openai.api_key = api_key

    classifications = []

    # Define the batch prompt template
    prompt_template = (
        "You are an AI assistant, specialized in the Israeli-Palestinian conflict."
        "You are designed to classify posts of users from a social network to 'Pro-Israel', 'Pro-Palestine' or 'Neutral' post."
        "You are not programmed to take sides or express opinions. Your goal is to provide accurate classification"
        "to help us understand the side of the post's writers about the conflict."
        "Let's use numeric labels for this task: 0 for Neutral, 1 for Pro-Israel and -1 for Pro-Palestine."
        "Here are the comments:\n"
        "{comments}\n\n"
        "Provide your response as a JSON list of integers corresponding to the classification of each comment. "
        "DO NOT include any other information in your response!!!!!"
    )

    # Process comments in batches
    for i in range(0, len(df), batch_size):
        batch = df['self_text'][i:i + batch_size].tolist()
        formatted_comments = "\n".join(f"{idx + 1}: {comment}" for idx, comment in enumerate(batch))

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt_template.format(comments=formatted_comments)}
                ]
            )

            # Extract the classifications
            classification = eval(response['choices'][0]['message']['content'].strip())  # Convert JSON string to Python list
            classifications.extend(classification)

        except Exception as e:
            print(f"Error classifying batch starting at index {i}. Error: {e}")
            classifications.extend([None] * len(batch))  # Handle errors

    # Save results to a new CSV
    df['classification'] = classifications
    df.to_csv(output_csv, index=False)
    print(f"Classifications saved to {output_csv}")


def evaluate_performance(classified_csv, actual_csv):
    # Load both CSV files
    classified_df = pd.read_csv(classified_csv)
    actual_df = pd.read_csv(actual_csv)

    if 'classification' not in classified_df.columns or 'Label' not in actual_df.columns:
        raise ValueError("Classified CSV must have a 'classification' column and Actual CSV must have a 'Label' column.")

    # Ensure alignment of data
    if len(classified_df) != len(actual_df):
        raise ValueError("The number of rows in the classified and actual CSV files must match.")

    # Calculate performance metrics
    y_pred = classified_df['classification'].fillna(0).astype(int)  # Handle None values
    y_true = actual_df['Label'].astype(int)

    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=["Pro-Palestine (-1)", "Neutral (0)", "Pro-Israel (1)"])
    print("\nPerformance Evaluation:")
    print(report)

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1])

    # Plot the confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Pro-Palestine (-1)", "Neutral (0)", "Pro-Israel (1)"],
                yticklabels=["Pro-Palestine (-1)", "Neutral (0)", "Pro-Israel (1)"])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    plt.show()


# open file contain OpenAI API key (txt file)
with open("api_key.txt") as f:
    api_key = f.read
# Usage example:
classify_comments_with_gpt(api_key, "test_data.csv", "classified_output_few_shot.csv", batch_size=25)
evaluate_performance("classified_output_few_shot.csv", "test_data.csv")
