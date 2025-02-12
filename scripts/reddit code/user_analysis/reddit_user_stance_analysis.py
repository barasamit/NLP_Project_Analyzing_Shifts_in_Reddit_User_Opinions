import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from collections import defaultdict
import glob
import os
from tqdm import tqdm


def load_and_process_data(dir_path, pattern="labeled_cleaned_IsraelPalestine_*_full_data.json"):
    """
    Load and process multiple JSON files containing Reddit data from a specified directory.

    Parameters:
        dir_path (str): Path to the directory containing the JSON files
        pattern (str): Pattern to match the JSON files

    Returns:
        DataFrame with user stances over time
    """
    all_data = []
    file_pattern = os.path.join(dir_path, pattern)

    # Get list of files first
    files = list(glob.glob(file_pattern))
    print(f"Found {len(files)} files to process")

    # Load all matching JSON files with progress bar
    for file_path in tqdm(files, desc="Processing files"):
        with open(file_path, 'r') as file:
            data = json.load(file)
            # Show progress for submissions within each file
            for submission_id, submission_data in tqdm(data.items(),
                                                       desc=f"Processing submissions in {os.path.basename(file_path)}",
                                                       leave=False):
                # Process submission
                submission = submission_data['submission']
                all_data.append({
                    'author': submission['author'],
                    'created_utc': pd.to_datetime(submission['created_utc']),
                    'stance': submission['stance'],
                    'type': 'submission',
                    'file': os.path.basename(file_path)  # Add source file for debugging
                })

                # Process comments
                for comment_thread in submission_data['comments'].values():
                    for comment in comment_thread:
                        all_data.append({
                            'author': comment['author'],
                            'created_utc': pd.to_datetime(comment['created_utc']),
                            'stance': comment['stance'],
                            'type': 'comment',
                            'file': os.path.basename(file_path)  # Add source file for debugging
                        })

    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    return df


def analyze_user_stance_changes(df):
    """
    Analyze stance changes for each user over time.
    Returns DataFrames for individual user changes and aggregated changes.
    """
    # Sort by date and group by user
    print("Calculating daily user stances...")
    df = df.sort_values('created_utc')
    user_stances = df.groupby(['author', pd.Grouper(key='created_utc', freq='D')])['stance'].mean().reset_index()

    # Calculate stance changes
    user_changes = []
    print("Analyzing stance changes for each user...")
    for user in tqdm(user_stances['author'].unique(), desc="Processing users"):
        user_data = user_stances[user_stances['author'] == user].sort_values('created_utc')
        if len(user_data) > 1:  # Only include users with multiple posts
            initial_stance = user_data['stance'].iloc[0]
            final_stance = user_data['stance'].iloc[-1]
            total_change = final_stance - initial_stance
            num_posts = len(user_data)
            user_changes.append({
                'author': user,
                'total_change': total_change,
                'num_posts': num_posts,
                'stance_data': user_data
            })

    return user_changes


def plot_top_users_stance_changes(user_changes, n=3):
    """
    Plot stance changes over time for top n users with most significant changes.
    Includes a 30-day moving average.
    """
    # Sort users by absolute change and number of posts
    sorted_users = sorted(user_changes,
                          key=lambda x: (abs(x['total_change']), x['num_posts']),
                          reverse=True)
    top_users = sorted_users[:n]

    # Create plot
    plt.figure(figsize=(15, 10))
    for user_data in top_users:
        stance_data = user_data['stance_data']

        # Calculate 30-day moving average
        stance_data = stance_data.set_index('created_utc')
        moving_avg = stance_data['stance'].rolling(window='120D', min_periods=1).mean()

        # Plot both raw data and moving average
        # plt.scatter(stance_data.index,
        #             stance_data['stance'],
        #             alpha=0.3,
        #             s=30)
        plt.plot(moving_avg.index,
                 moving_avg,
                 label=f"{user_data['author']} (Δ={user_data['total_change']:.2f})",
                 linewidth=2)

    plt.title(f"Stance Changes Over Time - Top {n} Users with Most Change\n(120-day moving average)")
    plt.xlabel("Date")
    plt.ylabel("Stance (-1: Pro-Palestine, 1: Pro-Israel)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    return plt


# def plot_aggregated_stance_changes(df, top_n=1000):
#     """
#     Plot mean stance changes over time for top 1000 most active users.
#     Includes a 30-day moving average.
#     """
#     # Get top users by activity
#     user_activity = df['author'].value_counts()
#     top_users = user_activity.head(top_n).index
#
#     # Filter data for top users
#     top_users_data = df[df['author'].isin(top_users)]
#
#     # Calculate daily mean stance
#     daily_stance = top_users_data.groupby(pd.Grouper(key='created_utc', freq='D'))['stance'].agg(
#         ['mean', 'std', 'count']).reset_index()
#
#     # Calculate 30-day moving average and standard deviation
#     daily_stance = daily_stance.set_index('created_utc')
#     moving_avg = daily_stance['mean'].rolling(window='15D', min_periods=1).mean()
#     moving_std = daily_stance['std'].rolling(window='15D', min_periods=1).mean()
#
#     # Plot
#     plt.figure(figsize=(15, 8))
#
#     # Plot raw daily means with low alpha
#     plt.scatter(daily_stance.index,
#                 daily_stance['mean'],
#                 alpha=0.2,
#                 color='blue',
#                 s=20,
#                 label='Daily Mean')
#
#     # Plot moving average
#     plt.plot(moving_avg.index,
#              moving_avg,
#              color='blue',
#              linewidth=2,
#              label='30-day Moving Average')
#
#     # Add confidence interval for moving average
#     plt.fill_between(moving_avg.index,
#                      moving_avg - moving_std,
#                      moving_avg + moving_std,
#                      alpha=0.3,
#                      color='blue',
#                      label='±1 std dev')
#
#     plt.title(f"Average Stance Changes Over Time (Top {top_n} Most Active Users)\n15-day Moving Average")
#     plt.xlabel("Date")
#     plt.ylabel("Mean Stance (-1: Pro-Palestine, 1: Pro-Israel)")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#
#     return plt

def plot_aggregated_stance_changes(df, top_n=10000):
    """
    Plot mean stance changes over time for top 1000 most active users.
    Uses daily averages instead of moving average.
    """
    # Get top users by activity
    user_activity = df['author'].value_counts()
    top_users = user_activity.head(top_n).index

    # Filter data for top users
    top_users_data = df[df['author'].isin(top_users)]

    # Calculate daily mean stance
    daily_stance = top_users_data.groupby(pd.Grouper(key='created_utc', freq='D'))['stance'].agg(
        ['mean', 'std', 'count']).reset_index()

    # Remove days with no data
    daily_stance = daily_stance.dropna()

    # Plot
    plt.figure(figsize=(15, 8))

    # Plot mean line
    plt.plot(daily_stance['created_utc'],
             daily_stance['mean'],
             color='blue',
             linewidth=1,
             label='Daily Mean')

    # Add confidence interval
    plt.fill_between(daily_stance['created_utc'],
                     daily_stance['mean'] - daily_stance['std'],
                     daily_stance['mean'] + daily_stance['std'],
                     alpha=0.3,
                     color='blue',
                     label='±1 std dev')

    plt.title(f"Average Daily Stance (Top {top_n} Most Active Users)")
    plt.xlabel("Date")
    plt.ylabel("Mean Stance (-1: Pro-Palestine, 1: Pro-Israel)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    return plt


def plot_directional_stance_changes(user_changes, n=10):
    """
    Create two separate plots:
    1. Top n users who changed from pro-Israel to pro-Palestine
    2. Top n users who changed from pro-Palestine to pro-Israel
    Each plot includes a 120-day moving average.
    """
    # Separate users based on direction of change and minimum activity
    pro_pal_to_israel = []  # Users who shifted from pro-Palestine to pro-Israel
    pro_israel_to_pal = []  # Users who shifted from pro-Israel to pro-Palestine

    for user in user_changes:
        # Skip users with fewer than 10 posts
        if user['num_posts'] < 10:
            continue
        initial_stance = user['stance_data']['stance'].iloc[0]
        final_stance = user['stance_data']['stance'].iloc[-1]

        # Only consider significant initial stances (>0.2 or <-0.2)
        if abs(initial_stance) > 0.2:
            if initial_stance < 0 and final_stance > initial_stance:
                pro_pal_to_israel.append(user)
            elif initial_stance > 0 and final_stance < initial_stance:
                pro_israel_to_pal.append(user)

    # Sort both lists by magnitude of change
    pro_pal_to_israel.sort(key=lambda x: abs(x['total_change']), reverse=True)
    pro_israel_to_pal.sort(key=lambda x: abs(x['total_change']), reverse=True)

    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 20))

    # Plot pro-Palestine to pro-Israel shifts
    for user_data in pro_pal_to_israel[:n]:
        stance_data = user_data['stance_data']
        stance_data = stance_data.set_index('created_utc')
        moving_avg = stance_data['stance'].rolling(window='120D', min_periods=1).mean()

        ax1.plot(moving_avg.index,
                 moving_avg,
                 label=f"{user_data['author']} (Δ={user_data['total_change']:.2f})",
                 linewidth=2)

    ax1.set_title(f"Top {n} Users Shifting from Pro-Palestine to Pro-Israel\n(120-day moving average)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Stance (-1: Pro-Palestine, 1: Pro-Israel)")
    ax1.grid(True)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Plot pro-Israel to pro-Palestine shifts
    for user_data in pro_israel_to_pal[:n]:
        stance_data = user_data['stance_data']
        stance_data = stance_data.set_index('created_utc')
        moving_avg = stance_data['stance'].rolling(window='120D', min_periods=1).mean()

        ax2.plot(moving_avg.index,
                 moving_avg,
                 label=f"{user_data['author']} (Δ={user_data['total_change']:.2f})",
                 linewidth=2)

    ax2.set_title(f"Top {n} Users Shifting from Pro-Israel to Pro-Palestine\n(120-day moving average)")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Stance (-1: Pro-Palestine, 1: Pro-Israel)")
    ax2.grid(True)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    return plt

def main():
    # Set the directory path where your JSON files are located
    dir_path = "/Users/ormeiri/Desktop/NLP_Project_Analyzing_Shifts_in_Reddit_User_Opinions/data/Reddit data parsed/cleaned_and_labeled_reddit_files"  # Replace with your actual directory path

    # Load and process data
    df = load_and_process_data(dir_path)

    # Analyze user stance changes
    user_changes = analyze_user_stance_changes(df)

    # Plot directional stance changes
    # plot_directional_stance_changes(user_changes, n=5)
    # plt.savefig('directional_stance_changes.png')

    # Plot top 10 users with most change
    plot_top_users_stance_changes(user_changes, n=10)
    plt.savefig('top_users_stance_changes.png')

    # Plot aggregated changes for top 1000 users
    # plot_aggregated_stance_changes(df, top_n=10000)
    # plt.savefig('aggregated_stance_changes.png')


if __name__ == "__main__":
    main()