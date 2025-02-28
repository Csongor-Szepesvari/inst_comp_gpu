import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

# combined data headers
### pct_total,game_mode,win_value_underdog,blind_combo_0,blind_combo_1,level_0,level_1,lognormal,pct_high_mean,high_low_ratio_mean,high_low_ratio_variance,mean_variance_ratio,pct_high_sigma,underdog_mean,underdog_variance ###

# Read the combined data
df = pd.read_csv("combined.csv")

# Filter for rows where underdog_mean > 0.5
filtered_df = df[df['underdog_mean'] > 0.5]

# Display results
total_rows = len(df)
filtered_rows = len(filtered_df)
print(f"Total number of rows in dataset: {total_rows}")
print(f"Number of rows where underdog_mean > 0.5: {filtered_rows}")
print(f"Percentage: {(filtered_rows/total_rows)*100:.2f}%")

# Further filter for rows where underdog_mean is between 0.5 and 0.53
filtered_df_below_53 = filtered_df[filtered_df['underdog_mean'] < 0.53]

# Display results for this additional filter
filtered_rows_below_53 = len(filtered_df_below_53)
print(f"\nOf those rows where underdog_mean > 0.5:")
print(f"Number of rows where underdog_mean < 0.53: {filtered_rows_below_53}")
print(f"Percentage of original dataset: {(filtered_rows_below_53/total_rows)*100:.2f}%")
print(f"Percentage of rows where underdog_mean > 0.5: {(filtered_rows_below_53/filtered_rows)*100:.2f}%")

# in ~57% of cases where we're beating the favourite, it's really close

# now let's create a filter that checks for the number of times the favourite gets a negative value
# Filter for rows where underdog_mean > 1
filtered_df_above_1 = df[df['underdog_mean'] > 1]

# Display results
filtered_rows_above_1 = len(filtered_df_above_1)
print(f"\nNumber of rows where the favourite has negative score: {filtered_rows_above_1}")
print(f"Percentage of original dataset: {(filtered_rows_above_1/total_rows)*100:.2f}%")



# and one where we get a negative value
# Filter for rows where underdog_mean is negative
filtered_df_negative = df[df['underdog_mean'] < 0]

# Display results
filtered_rows_negative = len(filtered_df_negative)
print(f"\nNumber of rows where underdog_mean is negative: {filtered_rows_negative}")
print(f"Percentage of original dataset: {(filtered_rows_negative/total_rows)*100:.2f}%")

# shockingly high numbers here

# let's investigate, let's break it down between top-k vs all and log vs normal
# Split negative cases by lognormal and game_mode
log_negative = filtered_df_negative[filtered_df_negative['lognormal'] == 'log']
normal_negative = filtered_df_negative[filtered_df_negative['lognormal'] == 'normal']

top_k_negative = filtered_df_negative[filtered_df_negative['game_mode'] == 'top_k']
expected_negative = filtered_df_negative[filtered_df_negative['game_mode'] == 'expected']

# Split above 1 cases by lognormal and game_mode
log_above_1 = filtered_df_above_1[filtered_df_above_1['lognormal'] == 'log']
normal_above_1 = filtered_df_above_1[filtered_df_above_1['lognormal'] == 'normal']

top_k_above_1 = filtered_df_above_1[filtered_df_above_1['game_mode'] == 'top_k']
expected_above_1 = filtered_df_above_1[filtered_df_above_1['game_mode'] == 'expected']

# Display results for negative cases
print("\nBreakdown of negative underdog_mean cases:")
print(f"Number of log cases: {len(log_negative)}")
print(f"Number of normal cases: {len(normal_negative)}")
print(f"Number of top_k cases: {len(top_k_negative)}")
print(f"Number of expected cases: {len(expected_negative)}")

# Display results for above 1 cases
print("\nBreakdown of underdog_mean > 1 cases:")
print(f"Number of log cases: {len(log_above_1)}")
print(f"Number of normal cases: {len(normal_above_1)}")
print(f"Number of top_k cases: {len(top_k_above_1)}")
print(f"Number of expected cases: {len(expected_above_1)}")

# Calculate percentages for negative cases
print("\nPercentages of negative cases:")
print(f"Log cases: {(len(log_negative)/filtered_rows_negative)*100:.2f}%")
print(f"Normal cases: {(len(normal_negative)/filtered_rows_negative)*100:.2f}%")
print(f"Top_k cases: {(len(top_k_negative)/filtered_rows_negative)*100:.2f}%")
print(f"Expected cases: {(len(expected_negative)/filtered_rows_negative)*100:.2f}%")

# Calculate percentages for above 1 cases
print("\nPercentages of underdog_mean > 1 cases:")
print(f"Log cases: {(len(log_above_1)/filtered_rows_above_1)*100:.2f}%")
print(f"Normal cases: {(len(normal_above_1)/filtered_rows_above_1)*100:.2f}%")
print(f"Top_k cases: {(len(top_k_above_1)/filtered_rows_above_1)*100:.2f}%")
print(f"Expected cases: {(len(expected_above_1)/filtered_rows_above_1)*100:.2f}%")

# Create a matrix breakdown for negative cases
print("\nMatrix breakdown of negative underdog_mean cases:")
negative_matrix = pd.crosstab(index=filtered_df_negative['lognormal'],columns=filtered_df_negative['game_mode'], margins=True,margins_name='Total')
print(negative_matrix)

# Create a matrix breakdown for above 1 cases
print("\nMatrix breakdown of underdog_mean > 1 cases:")
above_1_matrix = pd.crosstab(index=filtered_df_above_1['lognormal'], columns=filtered_df_above_1['game_mode'], margins=True, margins_name='Total')
print(above_1_matrix)

# Calculate percentage matrices
print("\nPercentage matrix of negative cases:")
negative_percent_matrix = negative_matrix / filtered_rows_negative * 100
print(negative_percent_matrix)

print("\nPercentage matrix of underdog_mean > 1 cases:")
above_1_percent_matrix = above_1_matrix / filtered_rows_above_1 * 100
print(above_1_percent_matrix)

# In what scenarios do we go negative? Let's make the same type of cross-tabulations with percentage_high_mean and total_occupancy
# Create crosstab for negative cases with pct_total and high_low_ratio_mean
print("\nNegative cases breakdown by pct_total and high_low_ratio_mean:")
negative_pct_ratio_matrix = pd.crosstab(
    index=filtered_df_negative['pct_total'],
    columns=filtered_df_negative['high_low_ratio_mean'], 
    margins=True,
    margins_name='Total'
)
print(negative_pct_ratio_matrix)

# Create crosstab for above 1 cases with pct_total and high_low_ratio_mean
print("\nAbove 1 cases breakdown by pct_total and high_low_ratio_mean:")
above_1_pct_ratio_matrix = pd.crosstab(
    index=filtered_df_above_1['pct_total'],
    columns=filtered_df_above_1['high_low_ratio_mean'],
    margins=True,
    margins_name='Total'
)
print(above_1_pct_ratio_matrix)

# Calculate percentage matrices
print("\nPercentage matrix of negative cases by pct_total and high_low_ratio_mean:")
negative_pct_ratio_percent_matrix = negative_pct_ratio_matrix / filtered_rows_negative * 100
print(negative_pct_ratio_percent_matrix)

print("\nPercentage matrix of above 1 cases by pct_total and high_low_ratio_mean:")
above_1_pct_ratio_percent_matrix = above_1_pct_ratio_matrix / filtered_rows_above_1 * 100
print(above_1_pct_ratio_percent_matrix)


# Analyze underdog_variance for cases where underdog_mean > 1
print("\nUnderdog_variance statistics for cases where underdog_mean > 1:")
above_1_variance_stats = filtered_df_above_1['underdog_variance'].describe()
print(above_1_variance_stats)

# Analyze underdog_variance for cases where underdog_mean is negative
print("\nUnderdog_variance statistics for cases where underdog_mean is negative:")
negative_variance_stats = filtered_df_negative['underdog_variance'].describe()
print(negative_variance_stats)

# Filter for cases where variance is greater than 1
high_variance_cases = df[df['underdog_variance'] > 1]
print("\nCases where underdog_variance > 1:")
print(high_variance_cases)

# Create categories for underdog_mean
conditions = [
    (df['underdog_mean'] > 1),
    (df['underdog_mean'] < 0),
    (df['underdog_mean'] >= 0) & (df['underdog_mean'] <= 1)
]
choices = ['Mean > 1', 'Mean < 0', '0 <= Mean <= 1']
df['mean_category'] = np.select(conditions, choices, default='Unknown')

# Create crosstab of mean categories vs game_mode
mean_category_crosstab = pd.crosstab(
    index=df['mean_category'],
    columns=df['game_mode'],
    margins=True,
    margins_name='Total'
)
print("\nCrosstab of underdog_mean categories vs game_mode:")
print(mean_category_crosstab)

# Create percentage crosstab
mean_category_percent = mean_category_crosstab / len(df) * 100
print("\nPercentage crosstab of underdog_mean categories vs game_mode:")
print(mean_category_percent)





# Create boxplots to visualize the variance distributions
plt.figure(figsize=(12, 6))
plt.boxplot([filtered_df_above_1['underdog_variance'], filtered_df_negative['underdog_variance']], 
            labels=['Mean > 1', 'Mean < 0'])
plt.title('Comparison of Underdog Variance Distributions')
plt.ylabel('Underdog Variance')
plt.show()
