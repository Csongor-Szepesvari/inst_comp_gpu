'''
This file's job is to contain a single function that processes a single row from a dataframe that outlines one experiment

The tasks are as follows:
1. Generate all of the objects necessary from the row description
2. Find the equilibrium
3. Simulate game 1000 times
'''

from objects_gpu import Game, Player, Category
import cupy as cp
import time as t
import scipy.stats as stats

def process_row(row, verbose=False):
    '''
    Process row takes a row from a dataframe, runs an experiment, simulates the outcome and returns the results
    '''
    if verbose:
        print(row)
        print()

    start = t.time()  # Measure runtime for one experiment

    # *** TASK 1: Generate objects ***
    players_list = [
        Player(
            win_value=row[f'win_value_{i}'],
            blind=row[f'blind_{i}'],
            level=row[f'level_{i}'],
            name=f"Player_{i}"
        ) for i in range(2)  # Adjust range as needed for the number of players
    ]

    categories_dict = {
        f"Q{i+1}": Category(
            name=f"Q{i+1}",
            mu=row[f"mean_Q{i+1}"],
            sigma=row[f"std_Q{i+1}"],
            size=row[f"size_Q{i+1}"],
            log_or_normal="log" if row['lognormal'] else "normal"
        ) for i in range(4)  # Assuming 4 categories
    }

    if verbose:
        print()
        print("Population distribution, mean, std over categories:", [
            (category.size, category.mean, category.std) for category in categories_dict.values()
        ])

    size = sum(category.size for category in categories_dict.values())
    to_admit = int((row['pct_high_mean'] * row['pct_total'] * size) // 2)

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

    # *** TASK 2: Find the equilibrium ***
    game.find_strategies_iterated_br()

    if verbose:
        print("Equilibrium strategies:", game.get_strat_list())

    # *** TASK 3: Simulate and store results ***
    num_runs = 10000
    results = cp.zeros(num_runs)

    for i in range(num_runs):
        outcome = game.simulate_game()
        results[i] = outcome[players_list[0].name]['pct_total_util']

    mean = cp.mean(results).item()
    std = cp.std(results).item()

    end = t.time()

    if verbose:
        print()
        print("Evaluation:")
        print(f'{num_runs} runs resulted in a mean total utility achieved of {mean} and standard deviation of {std}.')
        print(f'Took {end - start} seconds to run one experiment with {num_runs} simulations.')

    return mean, std

# Uncomment and adjust the following for standalone testing
'''
import pandas as pd
import os

file_path = os.path.join(os.getcwd(), 'not_started/params_file.csv')
df = pd.read_csv(file_path)

for i in range(len(df)):
    print(process_row(df.iloc[i], verbose=True))
'''
