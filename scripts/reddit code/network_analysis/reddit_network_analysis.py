import json
import networkx as nx
import matplotlib.pyplot as plt
import os
import re
from datetime import datetime
from collections import Counter


def parse_reddit_data(file_path):
    """
    Load and parse Reddit JSON data
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def extract_date_from_filename(filename):
    """
    Extract month and year from filename like 'labeled_cleaned_IsraelPalestine_2023-09_full_data.json'
    """
    match = re.search(r'(\d{4}-\d{2})', filename)
    if match:
        return match.group(1)
    return None


def create_network(data):
    """
    Create a network graph from the Reddit data
    Returns the graph and a dictionary of comment counts per user
    """
    G = nx.Graph()
    comment_counts = Counter()

    for submission_id, thread in data.items():
        submission = thread['submission']

        # Count submission as a comment for the author
        comment_counts[submission['author']] += 1

        # Add submission author as node
        G.add_node(submission['author'],
                   stance=submission.get('stance', 0),
                   type='submission_author')

        # Process comments
        for comment_group in thread['comments'].values():
            for comment in comment_group:
                # Count the comment
                comment_counts[comment['author']] += 1

                # Add comment author as node
                G.add_node(comment['author'],
                           stance=comment.get('stance', 0),
                           type='comment_author')

                # Add edge between comment author and parent
                if comment['parent_id'].startswith('t3_'):
                    # Direct reply to submission
                    G.add_edge(comment['author'],
                               submission['author'],
                               stance=comment.get('stance', 0))
                else:
                    # Reply to another comment
                    parent_id = comment['parent_id'].split('_')[1]
                    for parent_comment in comment_group:
                        if parent_comment.get('id') == parent_id:
                            G.add_edge(comment['author'],
                                       parent_comment['author'],
                                       stance=comment.get('stance', 0))
                            break

    return G, comment_counts


def plot_network(G, comment_counts, output_dir, month_year):
    """
    Plot network graph and save to file with improved layout and readability
    """
    plt.figure(figsize=(20, 20))

    # Create layout with more space between nodes
    pos = nx.spring_layout(G, k=2.0, iterations=100)

    # Draw edges first (under nodes)
    edge_colors = []
    for _, _, data in G.edges(data=True):
        stance = data.get('stance', 0)
        if stance > 0:
            edge_colors.append('#ffcccc')  # Light red
        elif stance < 0:
            edge_colors.append('#cccfff')  # Light blue
        else:
            edge_colors.append('#eeeeee')  # Light gray

    nx.draw_networkx_edges(G, pos,
                           edge_color=edge_colors,
                           width=0.5,
                           alpha=0.3)

    # Draw nodes
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        stance = G.nodes[node].get('stance', 0)
        if stance > 0:
            node_colors.append('#ff4444')  # Brighter red
        elif stance < 0:
            node_colors.append('#4444ff')  # Brighter blue
        else:
            node_colors.append('#999999')  # Neutral gray

        # Node size based on number of comments
        # Scale the size: base size of 100 plus 50 per comment
        node_sizes.append(100 + (50 * comment_counts[node]))

    # Draw nodes with white border for better visibility
    nx.draw_networkx_nodes(G, pos,
                           node_color=node_colors,
                           node_size=node_sizes,
                           alpha=0.9,
                           linewidths=1,
                           edgecolors='white')

    # Add title and remove axes for cleaner look
    plt.title(f'Reddit Network - {month_year}\nNode size represents number of comments',
              fontsize=16, pad=20)
    plt.axis('off')

    # Save plot
    output_path = os.path.join(output_dir, f'network_{month_year}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Print network statistics
    print(f'Network Statistics for {month_year}:')
    print(f'Number of nodes: {len(G.nodes())}')
    print(f'Number of edges: {len(G.edges())}')
    print(f'Average degree: {sum(dict(G.degree()).values()) / len(G.nodes()):.2f}')
    print(f'Density: {nx.density(G):.4f}')
    print('Most active users (by comment count):')
    for user, count in sorted(comment_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f'  {user}: {count} comments')
    print('---')


def main(json_file_path, output_base_dir):
    """
    Main function to generate network visualization
    """
    # Extract month-year from filename
    month_year = extract_date_from_filename(os.path.basename(json_file_path))
    if not month_year:
        raise ValueError("Could not extract date from filename")

    # Load data and create network
    data = parse_reddit_data(json_file_path)
    G, comment_counts = create_network(data)

    if len(G.nodes()) > 0:  # Only create plot if network has nodes
        plot_network(G, comment_counts, output_dir, month_year)
        print(f'Generated network plot for {month_year}')
    else:
        print(f'No data available for {month_year}')


if __name__ == "__main__":
    # Example usage
    json_file = "/Users/ormeiri/Desktop/NLP_Project_Analyzing_Shifts_in_Reddit_User_Opinions/data/Reddit data parsed/cleaned_and_labeled_reddit_files/labeled_cleaned_IsraelPalestine_2023-10_full_data.json"
    output_dir = "/Users/ormeiri/Desktop/NLP_Project_Analyzing_Shifts_in_Reddit_User_Opinions/scripts/reddit code/network_analysis/networks_plots"

    main(json_file, output_dir)