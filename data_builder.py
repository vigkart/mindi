import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from joblib import dump
from sklearn.model_selection import train_test_split

# Which algos would you like to consider?
features_data_dir = r"features_data"
features_files = os.listdir(features_data_dir)
algo_list = []
for i in range(len(features_files)):
    num = features_files[i].split('_')[0]
    algo_list.append(num)

print(algo_list)

def read_features_csv(algo_num):
    filename = str(algo_num) + "_features.csv"
    path = os.path.join(features_data_dir, filename)
    df = pd.read_csv(path, index_col=0)
    df = df.dropna()
    pnl_Series, features_frame = df.iloc[:, 0], df.iloc[:, 1:]

    return features_frame, pnl_Series.to_frame()


X_init, y_init = read_features_csv(algo_list[0])

for algo_num in algo_list[1:]:
    features_df, pnl = read_features_csv(algo_num)
    X_init = X_init.join(features_df, how="outer")
    y_init = y_init.join(pnl, how="outer")

# print(X_init.shape)
# X_init = X_init.dropna()
# print(X_init.shape)

# print(y_init.shape)
# y_init = y_init.dropna()
# print(y_init.shape)

# Creating ground truth vectors (any profit with a zscore higher than 1, computed only on winning days)
y_classified = y_init.copy()
winning_days = y_init[y_init > 0]

for column in y_init.columns:
    series = winning_days[column].dropna()
    std = series.std()
    mean = series.mean()
    required_value = 0

    y_classified.loc[y_classified[column] <= required_value, column] = 0
    y_classified.loc[y_classified[column] > required_value, column] = 1

X_init = X_init.fillna(0)
y_init = y_init.fillna(0)


# Splitting into Training and Test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_init, y_classified, test_size=0.2, shuffle=False
)


# Normalizing the data
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_train = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)

X_test_scaled = scaler.transform(
    X_test
)  # Scaling test set using scaler from training set
X_test = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)


# Saving training data
classification_data_dir = r"classification_data"

scaler_path = os.path.join(
    classification_data_dir, "blackbox_classification_scaler.joblib"
)
dump(scaler, scaler_path)  # Saving x_train scaler for usage

X_train_path = os.path.join(classification_data_dir, "x_train.csv")
X_test_path = os.path.join(classification_data_dir, "x_test.csv")
y_train_path = os.path.join(classification_data_dir, "y_train.csv")
y_test_path = os.path.join(classification_data_dir, "y_test.csv")

X_train = X_train.fillna(0)
X_test = X_test.fillna(0)
y_train = y_train.fillna(0)
y_test = y_test.fillna(0)

X_train.to_csv(X_train_path, index=True, float_format="%.6f")
X_test.to_csv(X_test_path, index=True, float_format="%.6f")
y_train.to_csv(y_train_path, index=True)
y_test.to_csv(y_test_path, index=True)

print(f"X_train.shape: {X_train.shape}")
print(f"y_train.shape: {y_train.shape}")