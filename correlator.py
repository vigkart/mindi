import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import math

df = pd.read_csv("training_data/sample_output3.csv")
quantile_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # Currently using deciles
tile_data = pd.DataFrame(index=range(10), columns=df.columns[2:])


# Note, this calculates quantiles using all available data, meaning that for larger day intervals less data is used (because of this, r-values cannot be compared with each other)
for col in df.columns[2:]:
    stat_df = df[['pnl', col]]
    stat_df = stat_df.dropna()
    
    # Pearson linearity 
    spearman, spval = spearmanr(stat_df[col], stat_df['pnl'])
    pearson, ppval = pearsonr(stat_df[col], stat_df['pnl'])
    t = (pearson * math.sqrt(10 - 2))/math.sqrt(1-(pearson ** 2))
    print(f"stat: {col}| spearman: {spearman} & pval {spval}, pearson: {pearson} & pval: {ppval}")

    # Getting value delimiting each quantile
    percentiles = stat_df[col].quantile(quantile_list, interpolation='midpoint')
    percentiles = pd.concat([percentiles, pd.Series([float('inf')])]).reset_index(drop=True)

    # Calculating mean pnl for each quantile then adding it to heatmap
    lower_bound = float('-inf')
    for percentile in percentiles:
        tile = stat_df[(stat_df[col] >= lower_bound) & (stat_df[col] < percentile)]
        lower_bound = percentile

        # adding avg pnl for each percentile to heatmap
        tile_data[col][percentiles[percentiles == percentile].index[0]] = tile['pnl'].mean()

# Corellating tile_data
print("\ntile_data\n")
tile_data = tile_data.apply(pd.to_numeric, errors='coerce')
for col in tile_data:
    spearman, spval = spearmanr(tile_data[col], pd.Series(tile_data[col].index))
    pearson, ppval = pearsonr(tile_data[col], pd.Series(tile_data[col].index))
    t = (pearson * math.sqrt(10 - 2))/math.sqrt(1-(pearson ** 2))
    print(f"stat: {col}| spearman: {spearman} & pval {spval}, pearson: {pearson} & pval: {ppval}")

# Min-Max normalizing heatmap
heatmap_data = tile_data
for col in heatmap_data:
    heatmap_data[col] = (heatmap_data[col] - heatmap_data[col].min()) / (heatmap_data[col].max() - heatmap_data[col].min())

# Plotting
# sns.heatmap(heatmap_data.astype(float))
# plt.show()