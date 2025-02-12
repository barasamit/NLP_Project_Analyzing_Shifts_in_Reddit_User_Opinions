import zstandard
import json
import io
from collections import defaultdict
from datetime import datetime
import re
import os


def extract_date_from_filename(filename):
    """Extract date from RS_YYYY-MM.zst or RC_YYYY-MM.zst pattern"""
    match = re.search(r'[RS|RC]_(\d{4}-\d{2})', filename)
    if match:
        return match.group(1)
    return None


def convert_timestamp(timestamp):
    """Convert Unix timestamp to readable date"""
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')


def extract_important_data(item, is_submission=True):
    """
    Extract relevant data from submission or comment
    """
    common_data = {
        'author': item.get('author'),
        'author_fullname': item.get('author_fullname'),
        'created_utc': convert_timestamp(item.get('created_utc', 0)),
        'score': item.get('score', 0),
        'edited': convert_timestamp(item.get('edited', 0)) if isinstance(item.get('edited'), (int, float)) else False,
        'author_flair_text': item.get('author_flair_text'),
    }

    if is_submission:
        submission_data = {
            'title': item.get('title'),
            'selftext': item.get('selftext'),
            'num_comments': item.get('num_comments', 0),
            'upvote_ratio': item.get('upvote_ratio', 0),
            'link_flair_text': item.get('link_flair_text'),
        }
        return {**common_data, **submission_data}
    else:
        comment_data = {
            'body': item.get('body'),
            'parent_id': item.get('parent_id'),
            'link_id': item.get('link_id'),
            'is_submitter': item.get('is_submitter', False),
        }
        return {**common_data, **comment_data}


def process_reddit_monthly_data(submissions_path, comments_path, target_subreddit, output_dir='.'):
    """
    Process Reddit submissions and comments for a specific month
    """
    date_str = extract_date_from_filename(submissions_path)
    if not date_str:
        print("Could not extract date from filename. Expected pattern: RS_YYYY-MM.zst")
        return None, None, None

    submission_dict = {}
    user_activity = defaultdict(lambda: {'submissions': [], 'comments': []})

    print(f"Processing {date_str} submissions from r/{target_subreddit}...")

    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Process submissions
        dctx = zstandard.ZstdDecompressor()
        with open(submissions_path, 'rb') as compressed_file:
            reader = dctx.stream_reader(compressed_file)
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')

            for line in text_stream:
                try:
                    submission = json.loads(line)
                    if submission.get('subreddit', '').lower() == target_subreddit.lower():
                        submission_id = submission['id']
                        processed_submission = extract_important_data(submission, is_submission=True)

                        submission_dict[submission_id] = {
                            'submission': processed_submission,
                            'comments': defaultdict(list),
                            'comment_count': 0,
                        }

                        # Track user activity
                        author = processed_submission['author']
                        if author and author != '[deleted]':
                            user_activity[author]['submissions'].append({
                                'id': submission_id,
                                'title': processed_submission['title'],
                                'created_utc': processed_submission['created_utc'],
                                'score': processed_submission['score']
                            })

                        print(f"\nFound submission: {processed_submission['title']}")

                except json.JSONDecodeError:
                    continue

        print(f"\nFound {len(submission_dict)} submissions")

        # Process comments
        print("\nProcessing comments...")
        dctx = zstandard.ZstdDecompressor()
        with open(comments_path, 'rb') as compressed_file:
            reader = dctx.stream_reader(compressed_file)
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')

            for line in text_stream:
                try:
                    comment = json.loads(line)
                    link_id = comment.get('link_id', '')
                    if link_id:
                        submission_id = link_id.split('_')[1]
                        if submission_id in submission_dict:
                            processed_comment = extract_important_data(comment, is_submission=False)
                            parent_id = processed_comment['parent_id'].split('_')[1]

                            submission_dict[submission_id]['comments'][parent_id].append(processed_comment)
                            submission_dict[submission_id]['comment_count'] += 1

                            # Track user activity
                            author = processed_comment['author']
                            if author and author != '[deleted]':
                                user_activity[author]['comments'].append({
                                    'submission_id': submission_id,
                                    'comment_id': comment['id'],
                                    'created_utc': processed_comment['created_utc'],
                                    'body': processed_comment['body'],
                                    'score': processed_comment['score'],
                                    'is_submitter': processed_comment['is_submitter']
                                })

                except json.JSONDecodeError:
                    continue

        # Save the data files
        base_filename = f"{target_subreddit}_{date_str}"

        # Save submissions and comments
        full_data_path = f"{output_dir}/{base_filename}_full_data.json"
        with open(full_data_path, 'w', encoding='utf-8') as f:
            json.dump(submission_dict, f, indent=2)

        # Save user activity summary
        user_activity_path = f"{output_dir}/{base_filename}_user_activity.json"
        with open(user_activity_path, 'w', encoding='utf-8') as f:
            json.dump(user_activity, f, indent=2)

        # Generate user stats for this month
        user_stats = {
            author: {
                'total_submissions': len(data['submissions']),
                'total_comments': len(data['comments']),
                'first_activity': min([
                    *[s['created_utc'] for s in data['submissions']] +
                     [c['created_utc'] for c in data['comments']]
                ]) if data['submissions'] or data['comments'] else None,
                'last_activity': max([
                    *[s['created_utc'] for s in data['submissions']] +
                     [c['created_utc'] for c in data['comments']]
                ]) if data['submissions'] or data['comments'] else None,
            }
            for author, data in user_activity.items()
        }

        user_stats_path = f"{output_dir}/{base_filename}_user_stats.json"
        with open(user_stats_path, 'w', encoding='utf-8') as f:
            json.dump(user_stats, f, indent=2)

        print(f"\nProcessed data for {date_str} saved:")
        print(f"- Full data: {full_data_path}")
        print(f"- User activity: {user_activity_path}")
        print(f"- User stats: {user_stats_path}")
        print(f"Total unique users for {date_str}: {len(user_activity)}")

        return submission_dict, user_activity, user_stats

    except Exception as e:
        print(f"Error processing files: {str(e)}")
        return None, None, None


def process_all_months(base_dir, target_subreddit, output_dir='processed_data'):
    """
    Process all monthly data files from comments and submissions directories
    """
    # Define directories
    submissions_dir = os.path.join(base_dir, 'submissions')
    comments_dir = os.path.join(base_dir, 'comments')

    # Verify directories exist
    if not os.path.exists(submissions_dir) or not os.path.exists(comments_dir):
        print(f"Error: Please ensure both directories exist:\n{submissions_dir}\n{comments_dir}")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all submission files
    submission_files = sorted([f for f in os.listdir(submissions_dir) if f.startswith('RS_')])

    for submission_file in submission_files:
        date_str = extract_date_from_filename(submission_file)
        if not date_str:
            continue

        comment_file = f"RC_{date_str}.zst"
        comment_path = os.path.join(comments_dir, comment_file)
        submission_path = os.path.join(submissions_dir, submission_file)

        # Check if corresponding comment file exists
        if not os.path.exists(comment_path):
            print(f"Warning: No matching comment file found for {submission_file}")
            continue

        print(f"\nProcessing data for {date_str}")
        print(f"Submissions file: {submission_path}")
        print(f"Comments file: {comment_path}")

        process_reddit_monthly_data(
            submission_path,
            comment_path,
            target_subreddit,
            output_dir
        )


# Example usage
if __name__ == "__main__":
    base_directory = "/Users/ormeiri/Downloads/reddit"  # Directory containing comments and submissions folders
    subreddit = "IsraelPalestine"
    output_directory = "processed_data"

    process_all_months(base_directory, subreddit, output_directory)