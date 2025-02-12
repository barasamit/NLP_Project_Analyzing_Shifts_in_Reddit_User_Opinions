import json
import os
from typing import Dict, Any, List, Set
from pathlib import Path
from collections import defaultdict


class RedditCommentTree:
    def __init__(self):
        self.comments_by_id = {}  # Store all comments for easy lookup
        self.children = defaultdict(list)  # Store parent->child relationships

    def add_comment(self, comment: Dict[str, Any]):
        """Add a comment to the tree and update relationships."""
        comment_id = self._extract_id_from_fullname(comment.get('parent_id', ''))
        self.comments_by_id[comment_id] = comment
        self.children[comment_id].append(comment)

    def _extract_id_from_fullname(self, fullname: str) -> str:
        """Extract the ID part from Reddit's fullname format (t1_xxx, t3_xxx, etc)."""
        if not fullname:
            return ''
        parts = fullname.split('_')
        return parts[1] if len(parts) > 1 else fullname

    def count_total_comments(self) -> int:
        """Count total number of comments in the tree."""
        # Count unique comments by their content
        unique_comments = set()
        for comments_list in self.children.values():
            for comment in comments_list:
                # Create a tuple of identifying fields
                comment_key = (
                    comment.get('author', ''),
                    comment.get('body', ''),
                    comment.get('created_utc', ''),
                    comment.get('author_fullname', '')
                )
                unique_comments.add(comment_key)
        return len(unique_comments)


def build_comment_tree(comments_dict: Dict[str, List[Dict]]) -> RedditCommentTree:
    """Build a comment tree from the Reddit comments dictionary."""
    tree = RedditCommentTree()

    # Add all comments to the tree
    for comment_list in comments_dict.values():
        for comment in comment_list:
            tree.add_comment(comment)

    return tree


def count_all_comments(comments_obj: Dict) -> int:
    """
    Count all unique comments in a submission, handling the Reddit comment structure.
    Takes into account:
    - Comments might appear multiple times in different places
    - Parent-child relationships through parent_id
    - Both t1_ (comment) and t3_ (submission) prefixes
    """
    if not comments_obj:
        return 0

    tree = build_comment_tree(comments_obj)
    return tree.count_total_comments()


def filter_top_submissions(data: Dict[str, Any], top_n: int = 20) -> Dict[str, Any]:
    """
    Filter the data to keep only the top N most commented submissions.

    Args:
        data: Dictionary containing Reddit submissions and their comments
        top_n: Number of top commented submissions to keep (default: 150)

    Returns:
        Filtered dictionary containing only the top N most commented submissions
    """
    # Calculate actual comment counts for each submission
    submission_counts = []
    for submission_id, submission_data in data.items():
        comment_count = count_all_comments(submission_data['comments'])
        submission_counts.append((
            submission_id,
            comment_count,
            submission_data['submission'].get('title', 'No Title')  # For logging
        ))

    # Sort by comment count in descending order
    sorted_submissions = sorted(submission_counts, key=lambda x: x[1], reverse=True)

    # Take top N submission IDs
    top_submission_ids = [submission_id for submission_id, _, _ in sorted_submissions[:top_n]]

    # Create new dictionary with only top N submissions
    filtered_data = {
        submission_id: data[submission_id]
        for submission_id in top_submission_ids
        if submission_id in data  # Safety check
    }

    # Print some statistics about the top submissions
    print("\nTop 10 most commented submissions:")
    for submission_id, count, title in sorted_submissions[:10]:
        print(f"  - {title[:50]}{'...' if len(title) > 50 else ''}: {count} comments")

    return filtered_data


def process_reddit_file(input_path: str, output_dir: str) -> None:
    """
    Process a Reddit data file to keep only the top 150 most commented submissions.

    Args:
        input_path: Path to the input JSON file
        output_dir: Directory to save the output file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate output path
    input_filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, f"filtered_{input_filename}")

    try:
        # Read input file
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"\nProcessing {input_filename}")
        print(f"Original submissions: {len(data)}")

        # Sample check of comment structure for the first submission
        first_submission_id = next(iter(data))
        first_submission = data[first_submission_id]
        comment_count = count_all_comments(first_submission['comments'])
        print(f"\nSample submission ({first_submission['submission'].get('title', 'No Title')[:50]}...):")
        print(f"  Comment count: {comment_count}")

        # Filter data
        filtered_data = filter_top_submissions(data)

        # Write output file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)

        # Calculate and display final statistics
        original_comments = sum(count_all_comments(submission_data['comments'])
                                for submission_data in data.values())
        filtered_comments = sum(count_all_comments(submission_data['comments'])
                                for submission_data in filtered_data.values())

        print(f"\nFinal Statistics:")
        print(f"  Original submissions: {len(data)}")
        print(f"  Original comments: {original_comments}")
        print(f"  Filtered submissions: {len(filtered_data)}")
        print(f"  Filtered comments: {filtered_comments}")
        print(f"  Percentage of comments kept: {(filtered_comments / original_comments * 100):.1f}%")
        print(f"  Saved to: {output_path}")
        print("-" * 50)

    except Exception as e:
        print(f"Error processing {input_filename}: {str(e)}")
        raise


def process_directory():
    """Process all full_data.json files in the input directory."""
    # Define directories
    input_dir = "/Users/ormeiri/Desktop/NLP_Project_Analyzing_Shifts_in_Reddit_User_Opinions/data/Reddit data parsed/cleaned_and_labeled_reddit_files"
    output_dir = "/Users/ormeiri/Desktop/NLP_Project_Analyzing_Shifts_in_Reddit_User_Opinions/data/Reddit data parsed/Filtered_cleaned_reddit_files"

    # Get all full_data.json files
    input_path = Path(input_dir)
    full_data_files = list(input_path.glob("*full_data.json"))

    print(f"Found {len(full_data_files)} files to process")
    print("=" * 50)

    # Process each file
    for file_path in full_data_files:
        process_reddit_file(str(file_path), output_dir)

    print("\nProcessing complete!")


if __name__ == "__main__":
    process_directory()