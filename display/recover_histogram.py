# we recover the histogram from the polynomial coefficients and display it from a csv file
# Use plotly to display the histogram

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import plotly.express as px

def split_histogram_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Define the columns for environment specifications
    env_columns = [
        'pct_total', 'game_mode', 'win_value_underdog', 'blind_combo_0', 'blind_combo_1',
        'level_0', 'level_1', 'lognormal', 'pct_high_mean', 'high_low_ratio_mean',
        'high_low_ratio_variance', 'mean_variance_ratio', 'pct_high_sigma'
    ]
    
    # Extract environment specifications
    env_df = df[env_columns]
    
    # Extract histogram data (bin edges, bin counts, mean, std)
    histogram_df = df.drop(columns=env_columns)
    
    return env_df, histogram_df

def recover_histogram(file_path):
    # Split the data into environment and histogram DataFrames
    env_df, histogram_df = split_histogram_data(file_path)
    
    # Extract the first row of histogram data
    histogram_row = histogram_df.iloc[0].values
    
    # Split the row into bin edges and bin counts
    num_bins = (len(histogram_row) - 2) // 2  # Subtract 2 for mean and std
    bin_edges = histogram_row[:num_bins + 1]
    hist_values = histogram_row[num_bins + 1:-2]
    mean = histogram_row[-2]
    std = histogram_row[-1]
    
    # Reconstruct the histogram
    x = bin_edges[:-1]  # Use bin edges as x values (excluding the last edge)
    y = hist_values     # Use bin counts as y values
    
    # Normalize y values to percentages of total
    y = y / np.sum(y)
    
    # Display the histogram using Plotly
    fig = px.bar(x=x, y=y)
    fig.update_layout(
        title="Recovered Histogram",
        xaxis_title="Value",
        yaxis_title="Probability Density"
    )
    fig.show()
    
    return env_df, x, y, mean, std

def sample_from_distribution(x, y, num_samples=1000):
    """
    Sample from a distribution defined by bin edges and bin counts.

    Args:
        x (np.ndarray): Array of bin edges (length n+1).
        y (np.ndarray): Array of bin counts (length n).
        num_samples (int): Number of samples to generate.

    Returns:
        np.ndarray: Random samples drawn from the distribution.
    """
    # Normalize y to sum to 1
    y_normalized = y / np.sum(y)

    # Create the cumulative distribution function (CDF)
    cdf = np.cumsum(y_normalized)

    # Generate uniform random samples
    random_samples = np.random.rand(num_samples)

    # Use the CDF to map uniform samples to bin indices
    sampled_indices = np.searchsorted(cdf, random_samples)

    # Map bin indices to x values by sampling uniformly within each bin
    bin_widths = np.diff(x)
    sampled_x = x[sampled_indices] + np.random.rand(num_samples) * bin_widths[sampled_indices]

    return sampled_x

env_df, x, y, mean, std = recover_histogram('../finished/params_file_001_occupancy0.98_modeexpected.csv')
print("Environment Specifications:")
print(env_df.iloc[0])
print("\nHistogram Data:")
print(f"Bin Edges: {x}")
print(f"Bin Counts: {y}")
print(f"Mean: {mean}")
print(f"Std: {std}")

samples = sample_from_distribution(x, y, num_samples=10)
print("Random Samples:", samples)

