import json
import os
import glob


def remove_automoderator_comments(file_path):
    """
    Remove all AutoModerator comments from a JSON file and overwrite the original file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Process each post
    for post_id, post_data in data.items():
        # Process comments section
        comments = post_data.get('comments', {})
        for comment_id, comment_list in list(comments.items()):
            # Filter out AutoModerator comments
            filtered_comments = [
                comment for comment in comment_list
                if comment.get('author', '').lower() != 'automoderator'
            ]

            # Update comment count
            if filtered_comments:
                comments[comment_id] = filtered_comments
            else:
                # Remove the comment_id entry if no comments remain
                del comments[comment_id]

        # Update the total comment count
        total_comments = sum(len(comment_list) for comment_list in comments.values())
        post_data['comment_count'] = total_comments

    # Overwrite the original file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return file_path


def process_all_files(directory_path):
    """
    Process all JSON files in the directory
    """
    # Find all JSON files
    pattern = os.path.join(directory_path, '*.json')
    json_files = glob.glob(pattern)

    processed_files = []
    for file_path in json_files:
        try:
            processed_file = remove_automoderator_comments(file_path)
            processed_files.append(processed_file)
            print(f"Successfully processed and overwrote: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    return processed_files


# Usage example
if __name__ == "__main__":
    directory_path = '/Users/ormeiri/Desktop/Homework/data mining from data structure/project/Miki_project/processed_data/data_with_sentiment'
    processed_files = process_all_files(directory_path)
    print(f"\nProcessed and overwrote {len(processed_files)} files")