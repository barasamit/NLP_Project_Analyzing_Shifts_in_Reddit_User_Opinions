import os
import json
import re

# Directory containing the JSON files
input_dir = '/Users/ormeiri/Desktop/NLP_Project_Analyzing_Shifts_in_Reddit_User_Opinions/data/Reddit data parsed/Filtered_cleaned_reddit_files'

# Function to remove quoted text starting with '>'
def remove_quoted_text(content):
    # Use regex to remove any lines starting with '>'
    cleaned_content = re.sub(r'>.*', '', content)
    # Strip excess whitespace
    return cleaned_content.strip()

# Function to clean all posts and comments
def clean_data(data):
    cleaned_data = {}
    for post_id, post_content in data.items():
        cleaned_data[post_id] = {}

        # Clean submissions
        if 'submission' in post_content:
            cleaned_submission = post_content['submission']
            for field in ['title', 'selftext']:
                if field in cleaned_submission and cleaned_submission[field]:
                    cleaned_submission[field] = remove_quoted_text(cleaned_submission[field])
            cleaned_data[post_id]['submission'] = cleaned_submission

        # Clean comments
        if 'comments' in post_content:
            cleaned_comments = {}
            for comment_id, comment_list in post_content['comments'].items():
                cleaned_comments[comment_id] = [
                    {**comment, 'body': remove_quoted_text(comment['body'])}
                    for comment in comment_list if 'body' in comment
                ]
            cleaned_data[post_id]['comments'] = cleaned_comments
    return cleaned_data

# Iterate through all JSON files in the directory
for file_name in os.listdir(input_dir):
    if file_name.endswith('.json'):  # Process only JSON files
        file_path = os.path.join(input_dir, file_name)
        print(f"Processing: {file_path}")
        # Load the JSON data
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Clean the data
        cleaned_data = clean_data(data)

        # Overwrite the file with cleaned data
        with open(file_path, 'w') as file:
            json.dump(cleaned_data, file, indent=4)

print("Processing complete. All files cleaned.")