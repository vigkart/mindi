"""
Correlator.py is for spitting out several human-readable stats. 

First, it correlates the trailing stats with the pnl using pearson and spearman correlation
Next, it separates the states into quantiles and saves all of them in tile_data (sorts df into quantiles for each stat, and finds avg profit for each quantile)
Then, it correlates each stat's quantile value (avg pnl for that quantile) against the number of quantiles (quantiles are in increasing order) using pearson & spearman correlation
Finally, it normalizes the tile_data and saves it in heatmap_data and displays this as a heatmap.

Using all of this, the user can spot correlations between trailing stats and pnl using a variety of metrics.
"""
import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr
from scipy.stats import spearmanr


algo_num = 46
training_data_dir = r'training_data'
filename = str(algo_num) + '_features.csv'
filename = os.path.join(training_data_dir, filename)

df = pd.read_csv(filename)
quantile_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # Currently using deciles
tile_data = pd.DataFrame(index=range(10), columns=df.columns[2:])


# Note, this calculates quantiles using all available data, meaning that for larger day intervals less data is used (because of this, r-values cannot be compared with each other)
print('\nTrailing Stats Data\n')
for col in df.columns[2:]:
    stat_df = df[[f'{algo_num}_pnl', col]]
    stat_df = stat_df.dropna()
    
    # Correlation
    spearman, spval = spearmanr(stat_df[col], stat_df[f'{algo_num}_pnl'])
    pearson, ppval = pearsonr(stat_df[col], stat_df[f'{algo_num}_pnl'])
    col_spaces = 24 - len(col)
    col_name = col
    for space in range(col_spaces):
        col_name += ' '
    print(f"stat: {col_name} | spearman: {round(spearman,5) } & pval {round(spval, 5)}, pearson: {round(pearson, 5)} & pval: {round(ppval, 5)}")

    # Getting value delimiting each quantile
    percentiles = stat_df[col].quantile(quantile_list, interpolation='midpoint')
    percentiles = pd.concat([percentiles, pd.Series([float('inf')])]).reset_index(drop=True)

    # Calculating mean pnl for each quantile then adding it to heatmap
    lower_bound = float('-inf')
    for percentile in percentiles:
        tile = stat_df[(stat_df[col] >= lower_bound) & (stat_df[col] < percentile)]
        lower_bound = percentile

        # adding avg pnl for each percentile to heatmap
        tile_data[col][percentiles[percentiles == percentile].index[0]] = tile[f'{algo_num}_pnl'].mean()

# Corellating tile_data
print("\nQuantile Data\n")
tile_data = tile_data.apply(pd.to_numeric, errors='coerce')
for col in tile_data:
    spearman, spval = spearmanr(tile_data[col], pd.Series(tile_data[col].index))
    pearson, ppval = pearsonr(tile_data[col], pd.Series(tile_data[col].index))
    col_spaces = 24 - len(col)
    col_name = col
    for space in range(col_spaces):
        col_name += ' '
    print(f"stat: {col_name} | spearman: {round(spearman,5) } & pval {round(spval, 5)}, pearson: {round(pearson, 5)} & pval: {round(ppval, 5)}")

# Min-Max normalizing heatmap
heatmap_data = tile_data
for col in heatmap_data:
    heatmap_data[col] = (heatmap_data[col] - heatmap_data[col].min()) / (heatmap_data[col].max() - heatmap_data[col].min())

# Plotting
sns.heatmap(heatmap_data.astype(float))
plt.show()