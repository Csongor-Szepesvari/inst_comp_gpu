'''
This file's job is to contain a single function that processes a single row from a dataframe that outlines one experiment

The tasks are as follows:
1. Generate all of the objects necessary from the row description
2. Find the equilibrium
3. Simulate game 1000 times
4. Fit a probability distribution to estimate the outcomes
'''

from objects_gpu import Game, Player, Category
import cupy as cp
import numpy as np
import time as t
import scipy.stats as stats
from scipy.optimize import curve_fit

def polynomial_pdf(x, *coeffs):
    """Polynomial function constrained to be non-negative."""
    poly = np.polyval(coeffs, x)
    return np.maximum(poly, 0)  # Ensure non-negativity

def process_row(row, verbose=False, poly_degree=3):
    '''
    Process row takes a row from a dataframe, runs an experiment, simulates the outcome, and returns the results.
    It also fits a probability distribution function using polynomial regression.
    '''
    base_mu = 5
    base_sigma = 5 / row['mean_variance_ratio']
    base_candidates = 120
    if verbose:
        print(row)
        print()


    start = t.time()  # Measure runtime for one experiment

    # *** TASK 1: Generate objects ***
    players_list = [
        Player(
            win_value=row[f'win_value_underdog'] if i == 0 else 1-row[f'win_value_underdog'],
            blind=row[f'blind_combo_{i}'],
            level=row[f'level_{i}'],
            name=f"Player_{i}"
        ) for i in range(2)  # Adjust range as needed for the number of players
    ]

    categories_dict = {}
    for i in range(4):
        category_name = f"Q{i+1}"
        
        # Determine mu: for Q1 and Q2, scale by high_low_ratio_mean; otherwise, use base_mu directly.
        if i in [0, 1]:
            mu = base_mu * row['high_low_ratio_mean']
        else:
            mu = base_mu
        
        # Determine sigma: for Q1 and Q3, scale by high_low_ratio_variance; otherwise, use base_sigma.
        if i in [0, 2]:
            sigma = base_sigma * row['high_low_ratio_variance']
        else:
            sigma = base_sigma
        
        # Determine size based on:
        # Q1: base_candidates * pct_high_mean * pct_high_sigma
        # Q2: base_candidates * pct_high_mean * (1 - pct_high_sigma)
        # Q3: base_candidates * (1 - pct_high_mean) * pct_high_sigma
        # Q4: base_candidates * (1 - pct_high_mean) * (1 - pct_high_sigma)
        if i == 0:
            size = int(base_candidates * row['pct_high_mean'] * row['pct_high_sigma'])
        elif i == 1:
            size = int(base_candidates * row['pct_high_mean'] * (1 - row['pct_high_sigma']))
        elif i == 2:
            size = int(base_candidates * (1 - row['pct_high_mean']) * row['pct_high_sigma'])
        else:  # i == 3
            size = int(base_candidates * (1 - row['pct_high_mean']) * (1 - row['pct_high_sigma']))
        
        log_or_normal = "log" if row['lognormal'] == 'log' else "normal"
        
        categories_dict[category_name] = Category(
            name=category_name,
            mu=mu,
            sigma=sigma,
            size=size,
            log_or_normal=log_or_normal
        )

    size = sum(category.size for category in categories_dict.values())
    print(size)
    to_admit = min(size//2,int((row['pct_high_mean'] * row['pct_total'] * size) // 2))

    top_k = None if row['game_mode'] == 'expected' else int(to_admit * 0.2)

    game = Game(
        num_players=len(players_list),
        to_admit=to_admit,
        players=players_list,
        categories=categories_dict,
        game_mode_type=row['game_mode'],
        top_k=top_k,
        log_normal=row['lognormal'],
        verbose=verbose
    )
    print([category.mean for category in categories_dict.values()])
    # *** TASK 2: Find the equilibrium ***
    game.find_strategies_iterated_br()

    print("Underdog strategy:")
    print(game.players[0].strategy)
    print("Favorite strategy:")
    print(game.players[1].strategy)

    # *** TASK 3: Simulate and store results ***
    num_runs = 10
    # Run all game simulations concurrently on the GPU by passing the total number of runs
    results = game.simulate_game_batch(num_runs)



    # Convert results to NumPy
    results_np = cp.asnumpy(results)

    # Estimate empirical density (normalized histogram)
    hist_values, bin_edges = np.histogram(results_np, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Fit a polynomial to the empirical density function
    poly_coeffs, _ = curve_fit(polynomial_pdf, bin_centers, hist_values, p0=np.ones(poly_degree + 1))
    # Poly_coeffs is a list of the coefficients of the polynomial used to fit the data

    mean = np.mean(results_np)
    std = np.std(results_np)

    end = t.time()


    if verbose:
        print()
        print("Evaluation:")
        print(f'{num_runs} runs resulted in a fitted probability distribution with polynomial coefficients: {poly_coeffs}')
        print(f'Took {end - start} seconds to run one experiment with {num_runs} simulations.')

    return poly_coeffs, mean, std


# Test the function by importing a single file from the not_started folder
# Run the function on a single row of the file
# Print the results

if __name__ == "__main__":
    import pandas as pd
    # Import a single file from the not_started folder
    
    # set up the file path by importing with os listdir
    import os
    file_path = 'not_started/' + os.listdir("not_started")[0]
    df = pd.read_csv(file_path)
    # Needs to get the second row of the dataframe because otherwise it's the header
    # print the header
    
    print(df.columns)
    row = df.iloc[1]
    print(row)
    #print()
    results = process_row(row)
    #print(results)


