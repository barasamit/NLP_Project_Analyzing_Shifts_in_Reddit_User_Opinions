import json
import os
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_stances_in_directory(directory_path: str) -> Dict:
    """
    Analyze stance distributions in all JSON files within a directory.

    Args:
        directory_path: Path to directory containing JSON files

    Returns:
        Dictionary containing aggregated stance statistics
    """
    # Initialize counters for different types of submissions
    stance_distribution = {
        1: {'pro_israel': 0, 'neutral': 0, 'pro_palestine': 0, 'total_comments': 0},  # Pro-Israel submissions
        0: {'pro_israel': 0, 'neutral': 0, 'pro_palestine': 0, 'total_comments': 0},  # Neutral submissions
        -1: {'pro_israel': 0, 'neutral': 0, 'pro_palestine': 0, 'total_comments': 0}  # Pro-Palestine submissions
    }

    submission_counts = {1: 0, 0: 0, -1: 0}  # Track number of submissions of each type

    # Process each JSON file in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"\nProcessing file: {filename}")

                # Process each submission in the file
                for submission_id, content in data.items():
                    submission = content.get('submission', {})
                    submission_stance = submission.get('stance')

                    # Skip if submission stance is not valid
                    if submission_stance not in [1, 0, -1]:
                        print(f"Skipping submission {submission_id} - invalid stance: {submission_stance}")
                        continue

                    submission_counts[submission_stance] += 1

                    # Process comments for this submission
                    comments = []
                    for comment_group in content.get('comments', {}).values():
                        if isinstance(comment_group, list):
                            comments.extend(comment_group)

                    # Count comment stances
                    valid_comments = 0
                    for comment in comments:
                        comment_stance = comment.get('stance')
                        if comment_stance in [1, 0, -1]:
                            if comment_stance == 1:
                                stance_distribution[submission_stance]['pro_israel'] += 1
                            elif comment_stance == 0:
                                stance_distribution[submission_stance]['neutral'] += 1
                            elif comment_stance == -1:
                                stance_distribution[submission_stance]['pro_palestine'] += 1
                            valid_comments += 1

                    stance_distribution[submission_stance]['total_comments'] += valid_comments

            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")

    print("\nSubmission counts:", submission_counts)
    return stance_distribution


def create_visualization(stance_distribution: Dict):
    """
    Create visualizations for the stance distribution data.

    Args:
        stance_distribution: Dictionary containing stance statistics
    """
    # Create percentage-based distribution
    data = []
    submission_types = {1: 'Pro-Israel', 0: 'Neutral', -1: 'Pro-Palestine'}

    # Print raw counts for debugging
    print("\nRaw counts before percentage calculation:")
    for stance, counts in stance_distribution.items():
        print(f"{submission_types[stance]}:", counts)

    for submission_stance, counts in stance_distribution.items():
        total = counts['total_comments']
        if total > 0:
            data.append({
                'Submission Type': submission_types[submission_stance],
                'Pro-Israel Comments %': (counts['pro_israel'] / total) * 100,
                'Neutral Comments %': (counts['neutral'] / total) * 100,
                'Pro-Palestine Comments %': (counts['pro_palestine'] / total) * 100
            })

    # Create DataFrame
    df = pd.DataFrame(data)
    print("\nPercentage DataFrame:")
    print(df)

    # Create stacked bar plot
    plt.figure(figsize=(12, 6))
    ax = df.plot(
        x='Submission Type',
        y=['Pro-Israel Comments %', 'Neutral Comments %', 'Pro-Palestine Comments %'],
        kind='bar',
        stacked=True,
        color=['#2563eb', 'gray', '#16a34a']
    )

    # Add percentage labels on each segment
    for c in ax.containers:
        # Add labels
        labels = [f'{v:.1f}%' if v >= 5 else '' for v in [v.get_height() for v in c]]
        ax.bar_label(c, labels=labels, label_type='center')

    plt.title('Comment Stance Distribution by Submission Type')
    plt.xlabel('Submission Type')
    plt.ylabel('Percentage of Comments')

    # Move legend outside of plot
    plt.legend(title='Comment Stances', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)

    # Save plot
    plt.savefig('stance_distribution.png')

    # Print numerical results
    print("\nNumerical Results:")
    for submission_stance, counts in stance_distribution.items():
        print(f"\n{submission_types[submission_stance]} Submissions:")
        total = counts['total_comments']
        if total > 0:
            print(f"Total comments: {total}")
            print(f"Pro-Israel comments: {counts['pro_israel']} ({counts['pro_israel'] / total * 100:.1f}%)")
            print(f"Neutral comments: {counts['neutral']} ({counts['neutral'] / total * 100:.1f}%)")
            print(f"Pro-Palestine comments: {counts['pro_palestine']} ({counts['pro_palestine'] / total * 100:.1f}%)")


def main(directory_path):
    """
    Main function to run the analysis.
    """

    # Validate directory path
    if not os.path.isdir(directory_path):
        print("Invalid directory path!")
        return

    # Run analysis
    print("Analyzing files...")
    stance_distribution = analyze_stances_in_directory(directory_path)
    print(stance_distribution)

    # Create visualization
    print("Creating visualization...")
    create_visualization(stance_distribution)

    print("\nAnalysis complete! Results have been saved to 'stance_distribution.png'")


if __name__ == "__main__":
    directory_path = "/Users/ormeiri/Desktop/NLP_Project_Analyzing_Shifts_in_Reddit_User_Opinions/data/Reddit data parsed/cleaned_and_labeled_reddit_files"
    main(directory_path)