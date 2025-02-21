# we recover the histogram from the polynomial coefficients and display it from a csv file
# Use plotly to display the histogram

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px





def recover_histogram(file_path, poly_degree=20):
    # read the csv file
    df = pd.read_csv(file_path)
    # get the polynomial coefficients
    coeffs = [str(i) for i in range(poly_degree+1)]
    poly_coeffs = df[coeffs].iloc[0].values
    # get the mean and std
    mean = df['mean'].values[0]
    std = df['std'].values[0]

    # recover the histogram
    x = np.linspace(0,1, num=1000)
    y = np.polyval(p=poly_coeffs, x=x)
    y = np.array([max(0,data) for data in y]) # lower bound it by 0
    print(len(y))
    # Rescale y values to percentages of total
    y = y / np.sum(y)
    # display the histogram
    
    fig = px.histogram(x=x, y=y, nbins=50)
    fig.show()
    return x, y

def sample_from_distribution(x, y, num_samples=1):
    """
    Randomly sample from a distribution defined by x and y, where y is the probability of x.

    Parameters:
        x (np.ndarray): Array of x values.
        y (np.ndarray): Array of y values representing the probability of x.
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

    # Use the CDF to map uniform samples to x values
    sampled_indices = np.searchsorted(cdf, random_samples)
    sampled_x = x[sampled_indices]

    return sampled_x

x, y = recover_histogram('params_file_001_occupancy0.8_modeexpected.csv')
samples = sample_from_distribution(x, y, num_samples=10)
print("Random Samples:", samples)

