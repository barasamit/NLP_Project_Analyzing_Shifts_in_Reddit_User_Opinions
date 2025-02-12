import json
import pandas as pd
from datetime import datetime
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.dates as mdates
import os


def load_cleaned_data(file_path):
    """Load cleaned and labeled JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_user_activities(data):
    """Convert user data to list of activities with timestamps and sentiments"""
    activities = []

    for activity_type in ['submissions', 'comments']:
        for activity in data.get(activity_type, []):
            activities.append({
                'type': activity_type,
                'timestamp': datetime.strptime(activity['created_utc'], '%Y-%m-%d %H:%M:%S'),
                'sentiment': activity['sentiment'],
                'id': activity.get('id') or activity.get('comment_id')
            })

    return activities


def analyze_sentiment_changes(base_path):
    """Analyze sentiment changes across all months"""
    all_user_activities = defaultdict(list)

    # Process each file in the directory
    for file_name in os.listdir(base_path):
        if file_name.startswith('cleaned_IsraelPalestine_') and file_name.endswith('_user_activity.json'):
            file_path = os.path.join(base_path, file_name)
            print(f"Processing {file_path}...")

            data = load_cleaned_data(file_path)

            # Aggregate user activities
            for username, user_data in data.items():
                activities = get_user_activities(user_data)
                all_user_activities[username].extend(activities)

    # Find top 500 most active users
    user_activity_counts = {
        username: len(activities)
        for username, activities in all_user_activities.items()
    }

    top_users = sorted(user_activity_counts.items(),
                       key=lambda x: x[1],
                       reverse=True)[:500]

    # Analyze sentiment changes for top users
    sentiment_changes = []

    for username, _ in top_users:
        activities = sorted(all_user_activities[username],
                            key=lambda x: x['timestamp'])

        if len(activities) < 10:  # Skip users with too few activities
            continue

        # Calculate daily sentiments first, then moving average
        df = pd.DataFrame(activities)
        df.set_index('timestamp', inplace=True)

        # Calculate 60-day moving average (approximately 2 months)
        moving_sentiments = df['sentiment'].rolling(window='60D', min_periods=5).mean()
        # Resample to get one point per day (to make the line smoother)
        moving_sentiments = moving_sentiments.resample('D').mean()
        # Remove days with no activity
        moving_sentiments = moving_sentiments.dropna()

        # Calculate sentiment change
        if len(moving_sentiments) >= 2:
            sentiment_change = {
                'username': username,
                'total_activities': len(activities),
                'start_sentiment': moving_sentiments.iloc[0],
                'end_sentiment': moving_sentiments.iloc[-1],
                'sentiment_change': moving_sentiments.iloc[-1] - moving_sentiments.iloc[0],
                'monthly_sentiments': moving_sentiments  # keeping the variable name for compatibility
            }
            sentiment_changes.append(sentiment_change)

    return sentiment_changes


def plot_sentiment_trends(sentiment_changes, output_path):
    """Plot sentiment trends for top 3 users with most significant changes"""
    # Sort by absolute sentiment change
    significant_changes = sorted(sentiment_changes,
                                 key=lambda x: abs(x['sentiment_change']),
                                 reverse=True)[:3]

    # Create plot with better styling
    plt.figure(figsize=(12, 8))

    # Set the style to minimal and clean
    plt.style.use('seaborn-v0_8-whitegrid')

    # Set background color
    plt.gca().set_facecolor('#f8f9fa')
    plt.gcf().set_facecolor('white')

    # Custom colors with better visibility
    colors = ['#2c3e50', '#e74c3c', '#27ae60']  # Dark blue, Red, Green

    for idx, user_data in enumerate(significant_changes):
        monthly_sentiments = user_data['monthly_sentiments']

        # Plot monthly moving average with improved line style
        plt.plot(monthly_sentiments.index,
                 monthly_sentiments.values,
                 color=colors[idx],
                 linewidth=2.5,
                 alpha=0.8,
                 label=f"{user_data['username']} (Δ={user_data['sentiment_change']:.2f})")

    # Improve title and labels
    plt.title('Sentiment Evolution of Most Changed Users\nSeptember 2023 - March 2024\n(60-day moving average)',
              fontsize=14,
              pad=20,
              fontweight='bold')

    plt.xlabel('Time Period', fontsize=12, labelpad=10)
    plt.ylabel('Sentiment Score', fontsize=12, labelpad=10)

    # Customize grid
    plt.grid(True, linestyle='--', alpha=0.7)

    # Improve legend
    plt.legend(bbox_to_anchor=(1.02, 1),
               loc='upper left',
               borderaxespad=0,
               frameon=True,
               fancybox=True,
               shadow=True)

    # Set y-axis limits with some padding
    plt.ylim(-1.1, 1.1)

    # Format x-axis
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%B %Y'))
    plt.xticks(rotation=45, ha='right')

    # Add subtle spines
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_color('#dddddd')

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save plot with higher quality
    plot_path = os.path.join(output_path, 'top3_monthly_sentiment_trends.png')
    plt.savefig(plot_path,
                dpi=300,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    plt.close()

    # Save detailed statistics for top 3 users
    stats_path = os.path.join(output_path, 'top3_users_stats.txt')
    with open(stats_path, 'w') as f:
        f.write("Top 3 Users with Most Significant Sentiment Changes:\n\n")
        for user_data in significant_changes:
            f.write(f"Username: {user_data['username']}\n")
            f.write(f"Total activities: {user_data['total_activities']}\n")
            f.write(f"Initial sentiment: {user_data['start_sentiment']:.2f}\n")
            f.write(f"Final sentiment: {user_data['end_sentiment']:.2f}\n")
            f.write(f"Total change: {user_data['sentiment_change']:.2f}\n")
            f.write(f"Average sentiment: {user_data['monthly_sentiments'].mean():.2f}\n")
            f.write(f"Sentiment standard deviation: {user_data['monthly_sentiments'].std():.2f}\n")
            f.write("\n")

    print(f"Plots and statistics saved to {output_path}")


def plot_average_sentiment_trend(sentiment_changes, output_path):
    """Plot the average sentiment trend for top 1000 users"""
    # Sort users by activity count and take top 1000
    top_1000_users = sorted(sentiment_changes,
                            key=lambda x: x['total_activities'],
                            reverse=True)[:1000]

    # Create a list to hold all sentiment series
    sentiment_series_list = []

    # Collect all sentiment series
    for user_data in top_1000_users:
        sentiments = user_data['monthly_sentiments']
        sentiment_series_list.append(pd.Series(sentiments, name=user_data['username']))

    # Combine all series at once using concat
    all_sentiments = pd.concat(sentiment_series_list, axis=1)

    # Calculate the mean sentiment across all users for each day
    mean_sentiment = all_sentiments.mean(axis=1)

    # Create plot with styling
    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn-v0_8-whitegrid')

    # Set background color
    plt.gca().set_facecolor('#f8f9fa')
    plt.gcf().set_facecolor('white')

    # Plot average sentiment
    plt.plot(mean_sentiment.index,
             mean_sentiment.values,
             color='#2c3e50',
             linewidth=3,
             alpha=0.8)

    # Add confidence interval
    std_sentiment = all_sentiments.std(axis=1)
    plt.fill_between(mean_sentiment.index,
                     mean_sentiment - std_sentiment,
                     mean_sentiment + std_sentiment,
                     color='#2c3e50',
                     alpha=0.2)

    # Improve title and labels
    plt.title(
        'Average Sentiment Trend of Top 1000 Most Active Users\nSeptember 2023 - March 2024\n(60-day moving average)',
        fontsize=14,
        pad=20,
        fontweight='bold')

    plt.xlabel('Time Period', fontsize=12, labelpad=10)
    plt.ylabel('Average Sentiment Score', fontsize=12, labelpad=10)

    # Customize grid
    plt.grid(True, linestyle='--', alpha=0.7)

    # Set y-axis limits with some padding
    plt.ylim(-1.1, 1.1)

    # Format x-axis
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%B %Y'))
    plt.xticks(rotation=45, ha='right')

    # Add subtle spines
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_color('#dddddd')

    # Adjust layout
    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_path, 'top1000_average_sentiment_trend.png')
    plt.savefig(plot_path,
                dpi=300,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    plt.close()

    # Save statistics
    stats_path = os.path.join(output_path, 'top1000_sentiment_stats.txt')
    with open(stats_path, 'w') as f:
        f.write("Average Sentiment Statistics for Top 1000 Users:\n\n")
        f.write(f"Overall mean sentiment: {mean_sentiment.mean():.3f}\n")
        f.write(f"Standard deviation: {mean_sentiment.std():.3f}\n")
        f.write(f"Maximum average sentiment: {mean_sentiment.max():.3f}\n")
        f.write(f"Minimum average sentiment: {mean_sentiment.min():.3f}\n")
        f.write(f"Total sentiment change: {mean_sentiment.iloc[-1] - mean_sentiment.iloc[0]:.3f}\n")

    print(f"Average sentiment trend plot and statistics saved to {output_path}")


def main():
    # Base path for cleaned and labeled data
    base_path = "/Users/ormeiri/Desktop/NLP_Project_Analyzing_Shifts_in_Reddit_User_Opinions/data/Reddit data parsed/cleaned_and_labeled_reddit_files"

    # Analyze sentiment changes
    sentiment_changes = analyze_sentiment_changes(base_path)

    # Plot trends and save results
    plot_sentiment_trends(sentiment_changes, base_path)

    # Plot average trend for top 1000 users
    plot_average_sentiment_trend(sentiment_changes, base_path)

    # Print summary statistics
    total_analyzed = len(sentiment_changes)
    positive_changes = sum(1 for x in sentiment_changes if x['sentiment_change'] > 0)
    negative_changes = sum(1 for x in sentiment_changes if x['sentiment_change'] < 0)

    print(f"\nOverall Statistics:")
    print(f"Total users analyzed: {total_analyzed}")
    print(f"Users with positive sentiment change: {positive_changes} ({positive_changes / total_analyzed * 100:.1f}%)")
    print(f"Users with negative sentiment change: {negative_changes} ({negative_changes / total_analyzed * 100:.1f}%)")


if __name__ == "__main__":
    main()