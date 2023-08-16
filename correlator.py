import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv("training_data/sample_output3.csv")
quantile_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # Currently using deciles
heatmap_data = pd.DataFrame(index=range(10), columns=df.columns[2:])



for col in df.columns[2:]:
    stat_df = df[['pnl', col]]
    stat_df = stat_df.dropna()
    
    # Getting value delimiting each quantile
    percentiles = stat_df[col].quantile(quantile_list, interpolation='midpoint')
    percentiles = percentiles.append(pd.Series([float('inf')]), ignore_index=True)
    
    # Calculating mean pnl for each quantile then adding it to heatmap
    lower_bound = float('-inf')
    for percentile in percentiles:
        tile = stat_df[(stat_df[col] >= lower_bound) & (stat_df[col] < percentile)]
        lower_bound = percentile

        # adding avg pnl for each percentile to heatmap
        heatmap_data[col][percentiles[percentiles == percentile].index[0]] = tile['pnl'].mean()


for col in heatmap_data.columns:
    heatmap_data[col] = (heatmap_data[col] - heatmap_data[col].min()) / (heatmap_data[col].max() - heatmap_data[col].min())


sns.heatmap(heatmap_data.astype(float))
plt.show()

# df['interval_start'] = pd.to_datetime(df['interval_start'])
# print(df)
# plt.plot(df['interval_start'], df['avg_pnl'])

# corr_df = df.drop(['interval_start', 'interval_end'], axis=1)

# corr = corr_df.astype('float').corr()
# heatmap = sns.heatmap(corr)

# plt.show()