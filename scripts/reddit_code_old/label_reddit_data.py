import json
import os
import glob
from textblob import TextBlob


def get_sentiment_score(text):
    """
    Analyze text sentiment and return -1, 0, or 1 based on polarity
    """
    if not text or text == "[removed]":
        return 0

    analysis = TextBlob(text)
    # Convert float polarity to discrete values
    if analysis.sentiment.polarity > 0.1:
        return 1
    elif analysis.sentiment.polarity < -0.1:
        return -1
    return 0


def process_json_file(file_path):
    """
    Process a single JSON file and add sentiment labels
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Process each post and its comments
    for post_id, post_data in data.items():
        # Analyze submission sentiment
        submission = post_data.get('submission', {})
        title_sentiment = get_sentiment_score(submission.get('title', ''))
        text_sentiment = get_sentiment_score(submission.get('selftext', ''))

        # Combine title and text sentiment
        submission['sentiment_label'] = title_sentiment if abs(title_sentiment) > abs(
            text_sentiment) else text_sentiment

        # Process comments
        comments = post_data.get('comments', {})
        for comment_id, comment_list in comments.items():
            for comment in comment_list:
                comment['sentiment_label'] = get_sentiment_score(comment.get('body', ''))

    # Save processed data
    output_path = file_path.replace('.json', '_with_sentiment.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return output_path


def process_all_files(directory_path):
    """
    Process all JSON files in the directory that contain 'full_data'
    """
    # Find all matching JSON files
    pattern = os.path.join(directory_path, '*full_data.json')
    json_files = glob.glob(pattern)

    processed_files = []
    for file_path in json_files:
        try:
            output_path = process_json_file(file_path)
            processed_files.append(output_path)
            print(f"Successfully processed: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    return processed_files


# Usage example
if __name__ == "__main__":
    directory_path = '/Users/ormeiri/Desktop/Homework/data mining from data structure/project/Miki_project/processed_data'
    processed_files = process_all_files(directory_path)
    print(processed_files)
    print(f"\nProcessed {len(processed_files)} files with sentiment labels")