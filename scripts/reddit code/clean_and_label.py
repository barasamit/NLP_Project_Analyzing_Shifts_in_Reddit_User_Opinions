import json
import pandas as pd
from datetime import datetime
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from pathlib import Path
import os

# Download required NLTK data
nltk.download('vader_lexicon')


def load_json_file(file_path):
    """Load and parse JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_sentiment_scores(text):
    """Calculate sentiment scores using VADER"""
    if not isinstance(text, str):
        return 0.0
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)['compound']


def process_user_activity_file(data):
    """Process user activity file, removing AutoModerator and calculating sentiments"""
    # Remove AutoModerator
    if 'AutoModerator' in data:
        del data['AutoModerator']

    # Process each user's data
    cleaned_data = {}
    for username, user_data in data.items():
        cleaned_user_data = {'submissions': [], 'comments': []}

        # Process submissions
        for submission in user_data.get('submissions', []):
            submission['sentiment'] = get_sentiment_scores(submission['title'])
            cleaned_user_data['submissions'].append(submission)

        # Process comments
        for comment in user_data.get('comments', []):
            comment['sentiment'] = get_sentiment_scores(comment['body'])
            cleaned_user_data['comments'].append(comment)

        cleaned_data[username] = cleaned_user_data

    return cleaned_data


def process_full_data_file(data):
    """Process full data file (submission tree with comments)"""
    cleaned_data = {}

    for submission_id, submission_data in data.items():
        # Get the submission info
        submission_info = submission_data.get('submission', {})

        # Skip if AutoModerator
        if submission_info.get('author') == 'AutoModerator':
            continue

        # Create cleaned submission data
        cleaned_submission = {
            'submission': {
                **submission_info,  # Keep all original submission data
                'sentiment': get_sentiment_scores(
                    submission_info.get('title', '') + ' ' + submission_info.get('selftext', ''))
            },
            'comments': {}
        }

        # Process comments
        comments_data = submission_data.get('comments', {})
        cleaned_comments = {}

        for comment_parent_id, comment_list in comments_data.items():
            cleaned_comment_list = []

            for comment in comment_list:
                # Skip if AutoModerator
                if comment.get('author') == 'AutoModerator':
                    continue

                # Add sentiment to comment
                comment_with_sentiment = {
                    **comment,  # Keep all original comment data
                    'sentiment': get_sentiment_scores(comment.get('body', ''))
                }
                cleaned_comment_list.append(comment_with_sentiment)

            if cleaned_comment_list:  # Only add if there are non-AutoModerator comments
                cleaned_comments[comment_parent_id] = cleaned_comment_list

        cleaned_submission['comments'] = cleaned_comments
        cleaned_data[submission_id] = cleaned_submission

    return cleaned_data


def process_user_stats_file(data):
    """Process user stats file"""
    # Remove AutoModerator
    if 'AutoModerator' in data:
        del data['AutoModerator']

    # No sentiment analysis needed for stats file, just remove AutoModerator
    return data


def process_file(input_file_path):
    """Process one file based on its type"""
    print(f"Processing {input_file_path}...")
    data = load_json_file(input_file_path)

    # Determine file type based on filename
    if 'user_activity' in input_file_path:
        return process_user_activity_file(data)
    elif 'full_data' in input_file_path:
        return process_full_data_file(data)
    elif 'user_stats' in input_file_path:
        return process_user_stats_file(data)
    else:
        raise ValueError(f"Unknown file type: {input_file_path}")


def main():
    # Input and output paths
    input_base_path = "/Users/ormeiri/Desktop/NLP_Project_Analyzing_Shifts_in_Reddit_User_Opinions/data/Reddit data parsed/processed_data"
    output_base_path = "/Users/ormeiri/Desktop/NLP_Project_Analyzing_Shifts_in_Reddit_User_Opinions/data/Reddit data parsed/cleaned_and_labeled_reddit_files"

    # Create output directory if it doesn't exist
    Path(output_base_path).mkdir(parents=True, exist_ok=True)

    # Process files for each month
    months = ['09', '10', '11', '12', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    years = ['2023', '2023', '2023', '2023', '2024', '2024', '2024', '2024', '2024', '2024', '2024', '2024', '2024', '2024']

    # File types to process
    # file_types = ['user_activity', 'full_data', 'user_stats']
    file_types = ['full_data']

    for year, month in zip(years, months):
        for file_type in file_types:
            # Construct file paths
            input_file = os.path.join(input_base_path, f'IsraelPalestine_{year}-{month}_{file_type}.json')
            output_file = os.path.join(output_base_path, f'cleaned_IsraelPalestine_{year}-{month}_{file_type}.json')

            # Process file if it exists
            if os.path.exists(input_file):
                try:
                    cleaned_data = process_file(input_file)

                    # Save cleaned and labeled data
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(cleaned_data, f, indent=2)
                    print(f"Saved cleaned and labeled data to {output_file}")

                except Exception as e:
                    print(f"Error processing {input_file}: {str(e)}")
            else:
                print(f"Warning: File not found - {input_file}")


if __name__ == "__main__":
    main()