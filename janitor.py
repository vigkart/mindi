import pandas as pd
import os
from scipy.stats import zscore
import numpy as np

# Define the directory paths
raw_data_dir = r"trading_data/raw_data"
clean_data_dir = r"trading_data/clean_data"
outliers_dir = r"trading_data/outliers"

# Create the output directories if they don't exist
os.makedirs(clean_data_dir, exist_ok=True)
os.makedirs(outliers_dir, exist_ok=True)

# Iterate over all CSV files in the raw data directory
for filename in os.listdir(raw_data_dir):
    if filename.endswith("_matches.csv"):
        # Extract the algorithm number from the filename
        algo_number = filename.split("_")[0]

        # Creating output paths
        clean_data_path = os.path.join(clean_data_dir, f"{algo_number}_clean.csv")
        outliers_path = os.path.join(outliers_dir, f"{algo_number}_outliers.csv")

        # If file has already been cleaned, skip
        if os.path.isfile(clean_data_path):
            print(f"Algo number {algo_number} has been processed already")
            continue

        # Load the data
        data_path = os.path.join(raw_data_dir, filename)
        data = pd.read_csv(data_path, keep_default_na=False)

        # Checking if turd to decide limit for cutting low value, unclean trades
        turd = [2, 7, 8, 9, 10, 11, 23]
        if int(algo_number) in turd:
            # Remove rows where 'avg_entry_px' or 'avg_exit_px' is < .50 for turd strategies
            data = data[
                (data["avg_entry_price"] > 0.5) | (data["avg_exit_price"] > 0.5)
            ]
        else:
            # Remove rows where 'avg_entry_px' or 'avg_exit_px' is < 5 for non-turd strategies
            data = data[(data["avg_entry_price"] > 5) | (data["avg_exit_price"] > 5)]

        # Calculate the price ratio
        data["price_ratio"] = data["avg_entry_price"] / data["avg_exit_price"]

        # Removing rows with any infinite values (which can be introduced in previous step)
        numeric_columns = data.select_dtypes(include=[float, int]).columns
        infinite_rows = data[
            data[numeric_columns].applymap(np.isinf).any(axis=1)
        ]  # infinite values are being squashed here (done after calculating price ratio because of potentially introduced inf's)
        data = data[~data[numeric_columns].applymap(np.isinf).any(axis=1)]

        # Calculating z-scores for avg_entry_px / avg_exit_px and pnl
        if data["price_ratio"].isna().any():
            print("NaN values introduced in price_ratio.")
        data["price_ratio_z_score"] = zscore(data["price_ratio"])
        data["pnl_z_score"] = zscore(data["pnl"])

        # Create the clean dataset by removing the outliers in both the price ratio and pnl
        clean_data = data[
            (abs(data["price_ratio_z_score"]) <= 3) & (abs(data["pnl_z_score"]) <= 3)
        ]

        # Identify the outliers
        outliers = data[
            (abs(data["price_ratio_z_score"]) > 3) & (abs(data["pnl_z_score"]) > 3)
        ]
        outliers = pd.concat([outliers, infinite_rows], ignore_index=True)

        # Save the clean data to a CSV file
        clean_data.to_csv(clean_data_path, index=False)

        # Save the outliers to a separate CSV file
        outliers.to_csv(outliers_path, index=False)
        print(f"Cleaned {filename}")
