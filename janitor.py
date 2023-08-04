import pandas as pd
import os
from scipy.stats import zscore

# Define the directory paths
raw_data_dir = r'trading_data/raw_data'
clean_data_dir = r'trading_data/clean_data'
outliers_dir = r'trading_data/outliers'

# Create the output directories if they don't exist
os.makedirs(clean_data_dir, exist_ok=True)
os.makedirs(outliers_dir, exist_ok=True)

# Iterate over all CSV files in the raw data directory
for filename in os.listdir(raw_data_dir):
    if filename.endswith('_matches.csv'):
        # Extract the algorithm number from the filename
        algo_number = filename.split('_')[0]

        # Load the data
        data_path = os.path.join(raw_data_dir, filename)
        data = pd.read_csv(data_path)
        
        # Checking if turd to decide limit for cutting low value, unclean trades
        turd = [2, 7, 8 , 9 , 10 , 11, 23]
        if int(algo_number) in turd:
            # Remove rows where 'avg_entry_px' or 'avg_exit_px' is < .50 for turd strategies
            data = data[(data['avg_entry_px'] > .5) | (data['avg_exit_px'] > .5)]
        else:
            # Remove rows where 'avg_entry_px' or 'avg_exit_px' is < 5 for non-turd strategies
            data = data[(data['avg_entry_px'] > 5) | (data['avg_exit_px'] > 5)]
        
        # Calculate the ratio and z-scores for avg_entry_px / avg_exit_px and pnl
        data['price_ratio'] = data['avg_entry_px'] / data['avg_exit_px']
        data['price_ratio_z_score'] = zscore(data['price_ratio'])
        data['pnl_z_score'] = zscore(data['pnl'])

        # Create the clean dataset by removing the outliers in both the price ratio and pnl
        clean_data = data[(abs(data['price_ratio_z_score']) <= 3) & (abs(data['pnl_z_score']) <= 3)]

        # Identify the outliers
        outliers = data[(abs(data['price_ratio_z_score']) > 3) & (abs(data['pnl_z_score']) > 3)]

        # Save the clean data to a CSV file
        clean_data_path = os.path.join(clean_data_dir, f'{algo_number}_clean.csv')
        clean_data.to_csv(clean_data_path, index=False)

        # Save the outliers to a separate CSV file
        outliers_path = os.path.join(outliers_dir, f'{algo_number}_outliers.csv')
        outliers.to_csv(outliers_path, index=False)