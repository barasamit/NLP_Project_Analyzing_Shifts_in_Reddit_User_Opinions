import json
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from pathlib import Path


def load_json_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def get_tree_size(submission_id, comments):
    if submission_id not in comments:
        return 0
    return len(comments[submission_id])


def create_network(submission, comments):
    G = nx.DiGraph()

    # Add submission node
    submission_sentiment = submission['sentiment']
    color = 'blue' if submission_sentiment > 0 else 'red' if submission_sentiment < 0 else 'gray'
    G.add_node(submission['author'], color=color, size=1000)

    # Add comment nodes and edges
    if submission['id'] in comments:
        for comment in comments[submission['id']]:
            sentiment = comment['sentiment']
            color = 'blue' if sentiment > 0 else 'red' if sentiment < 0 else 'gray'
            G.add_node(comment['author'], color=color, size=300)
            parent_id = comment['parent_id']

            if parent_id.startswith('t3_'):  # Direct reply to submission
                G.add_edge(comment['author'], submission['author'])
            else:  # Reply to comment
                parent_comment = next((c for c in comments[submission['id']]
                                       if c['id'] == parent_id[3:]), None)
                if parent_comment:
                    G.add_edge(comment['author'], parent_comment['author'])

    return G


def plot_network(G, output_path, submission_title):
    plt.figure(figsize=(15, 10))

    # Get node colors and sizes
    colors = [G.nodes[node]['color'] for node in G.nodes()]
    sizes = [G.nodes[node]['size'] for node in G.nodes()]

    # Spring layout for better visualization
    pos = nx.spring_layout(G, k=1, iterations=50)

    # Draw the network
    nx.draw(G, pos,
            node_color=colors,
            node_size=sizes,
            with_labels=True,
            font_size=8,
            edge_color='gray',
            arrows=True)

    plt.title(f"Network for: {submission_title[:50]}...")
    plt.savefig(output_path)
    plt.close()


def analyze_reddit_networks(pivot_date, data_dir, output_base_dir):
    # Convert pivot_date to datetime
    pivot_dt = datetime.strptime(pivot_date, '%Y-%m-%d')
    start_date = pivot_dt - timedelta(days=7)
    end_date = pivot_dt + timedelta(days=7)

    # Create output directory
    output_dir = os.path.join(output_base_dir, pivot_date)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get relevant month files
    months_needed = set()
    current_date = start_date
    while current_date <= end_date:
        months_needed.add(current_date.strftime('%Y-%m'))
        current_date += timedelta(days=1)

    submissions_in_range = []

    # Load and process data from each needed month
    for month in months_needed:
        file_path = os.path.join(data_dir, f'cleaned_IsraelPalestine_{month}_full_data.json')
        if not os.path.exists(file_path):
            continue

        data = load_json_data(file_path)

        # Filter submissions within date range
        for submission_id, submission_data in data.items():
            submission_date = datetime.strptime(
                submission_data['submission']['created_utc'],
                '%Y-%m-%d %H:%M:%S'
            )

            if start_date <= submission_date <= end_date:
                submissions_in_range.append({
                    'id': submission_id,
                    'data': submission_data,
                    'tree_size': get_tree_size(submission_id, submission_data['comments'])
                })

    # Sort by tree size and take top 10
    top_submissions = sorted(submissions_in_range,
                             key=lambda x: x['tree_size'],
                             reverse=True)[:10]

    # Create and save networks for top submissions
    for idx, submission in enumerate(top_submissions):
        G = create_network(
            {**submission['data']['submission'], 'id': submission['id']},
            submission['data']['comments']
        )

        submission_date = datetime.strptime(submission['data']['submission']['created_utc'], '%Y-%m-%d %H:%M:%S')
        output_path = os.path.join(output_dir, f'network_{submission_date.strftime("%Y-%m-%d")}_{idx + 1}.png')
        plot_network(G, output_path, submission['data']['submission']['title'])


# Example usage
if __name__ == "__main__":
    data_dir = "/Users/ormeiri/Desktop/NLP_Project_Analyzing_Shifts_in_Reddit_User_Opinions/data/Reddit data parsed/cleaned_and_labeled_reddit_files"
    output_base_dir = "/Users/ormeiri/Desktop/NLP_Project_Analyzing_Shifts_in_Reddit_User_Opinions/scripts/reddit code/network_analysis/networks_plots"

    # Example dates to analyze
    dates_to_analyze = [
        "2023-10-07",  # Example pivot date
    ]

    for date in dates_to_analyze:
        analyze_reddit_networks(date, data_dir, output_base_dir)