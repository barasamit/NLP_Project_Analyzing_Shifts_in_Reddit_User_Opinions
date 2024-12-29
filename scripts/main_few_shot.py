import openai
import pandas as pd
from sklearn.metrics import classification_report


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
        "You are designed to classify posts of users from a social network to 'Pro-Israel' ,'Pro-Palestine' or 'Neutral' post."
        "You are not programmed to take sides or express opinions. Your goal is to provide accurate classification"
        "to help us understand the side of the post's writers about the conflict."
        "Let's use numeric labels for this task: 0 for Neutral, 1 for Pro-Israel and -1 for Pro-Palestine."
        "These are some comments from a social network regarding the Israeli-Palestinian conflict:"
        "Neutural comment: Stealing land has been going on since humans decided one can own land unfortunately. The thing is you either pay now or you pay later but you always will pay more later. If we negotiated peace in 67 it would have been expensive but not as much as today and definitely not as much as tomorrow. Bite the bullet"
        "Pro-Israel: Israel is a wonderful country with wonderful people that have contributed a lot to the world. The model country for the Middle East (if they got along better with their neighbors). Your far right government and historically harsh policies towards Palestinians is a plague to your reputation that is only partially justified. The first step toward repairing it would be getting the politicians whoג€™ve had a grip on the political environment for decades, like Bibi, as far away from a position of power as possible.."
        "Pro-Palestine: The Hamas charter says that Islam will destroy Israel at some point, which even Donald Trump claims is true. It doesnt say anything about Hamas wanting to destroy Israel."

        "I will provide you with a dataset of user posts regarding the conflict,"
        "and you will classify each post according to the user's opinion."
        "So as a sentiment analysis expert, Classify the following comments into one of three categories: \n"
        "1: Pro-Israel\n"
        "0: Neutral\n"
        "-1: Pro-Palestine\n\n"
        "Here are the comments:\n"
        "{comments}\n\n"
        "Provide your response as a JSON list of integers corresponding to the classification of each comment. "
        "DO NOT include any other information in your response!!!!!"
        "For example: [1, 0, -1]."
        "DO NOT include any other information in your response!!!!!."
    )

    # Process comments in batches
    for i in range(0, len(df), batch_size):
        batch = df['self_text'][i:i + batch_size].tolist()
        formatted_comments = "\n".join(f"{idx + 1}: {comment}" for idx, comment in enumerate(batch))

        print('##############################################')

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4-1106-preview",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt_template.format(comments=formatted_comments)}
                ]
            )

            print("Response:")
            print(response['choices'][0]['message']['content'])
            # Extract the classifications
            classification = eval(
                response['choices'][0]['message']['content'].strip())  # Convert JSON string to Python list
            classifications.extend(classification)


        except Exception as e:
            print(f"Error classifying batch starting at index {i}. Error: {e}")
            classifications.extend([None] * len(batch))  # Handle errors

    # Save results to a new CSV
    df['classification'] = classifications
    df.to_csv(output_csv, index=False)
    print(f"Classifications saved to {output_csv}")


# Function to evaluate model performance
def evaluate_performance(classified_csv, actual_csv):
    # Load both CSV files
    classified_df = pd.read_csv(classified_csv)
    actual_df = pd.read_csv(actual_csv)

    if 'classification' not in classified_df.columns or 'Label' not in actual_df.columns:
        raise ValueError(
            "Classified CSV must have a 'classification' column and Actual CSV must have a 'Label' column.")

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

# open file contain OpenAI API key (txt file)
with open("api_key.txt") as f:
    api_key = f.read
# Usage example:
classify_comments_with_gpt(api_key, "test_data.csv", "classified_output_few_shot.csv", batch_size=25)
evaluate_performance("classified_output_few_shot.csv", "test_data.csv")



