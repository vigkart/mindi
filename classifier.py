import tensorflow as tf
import numpy as np
import pandas as pd
import os
from scipy.spatial import distance

# Getting the data
classification_data_dir = r"classification_data"

X_train_path = os.path.join(classification_data_dir, "x_train.csv")
X_test_path = os.path.join(classification_data_dir, "x_test.csv")
y_train_path = os.path.join(classification_data_dir, "y_train.csv")
y_test_path = os.path.join(classification_data_dir, "y_test.csv")

X_train = pd.read_csv(X_train_path, index_col=0)
X_test = pd.read_csv(X_test_path, index_col=0)
y_train = pd.read_csv(y_train_path, index_col=0)
y_test = pd.read_csv(y_test_path, index_col=0)

assert not np.any(np.isnan(X_train))
assert not np.any(np.isnan(y_train))

# Scaling layer shapes for given features and algorithms
nfeatures = X_train.shape[1]
nalgos = y_train.shape[1]
layer1_nodes = int(nfeatures/2)
layer2_nodes = int(layer1_nodes/2)
layer3_nodes = int(layer2_nodes/2)

np.random.seed(42)
tf.random.set_seed(42)
# initializer = tf.keras.initializers.HeNormal()
tf.keras.layers.Layer.default_kernel_initializer = tf.keras.initializers.HeNormal()

# Build the model
model = tf.keras.Sequential([
        tf.keras.layers.Dense(nfeatures, input_shape=(nfeatures,)),  # Define input shape for the first layer
        tf.keras.layers.BatchNormalization(),  # Batch normalization layer for the input
        tf.keras.layers.Dropout(.5),
        tf.keras.layers.Dense(layer1_nodes, activation="relu"),
        tf.keras.layers.Dropout(.5),
        tf.keras.layers.Dense(layer2_nodes, activation="relu"),
        tf.keras.layers.Dense(layer3_nodes, activation="relu"),
        tf.keras.layers.Dense(nalgos, activation="sigmoid")
    ]
)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=1.0)
model.compile(
    optimizer=optimizer,
    # loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=["accuracy"],
)

model.fit(X_train, y_train, epochs=64)

# Make the predictions
predictions = model.predict(X_test)
threshold = 0.5
predictions = pd.DataFrame(data=predictions, index=y_test.index, columns=y_test.columns)
threshold = 0.5
binary_predictions = (predictions > threshold).astype(int)
binary_predictions = binary_predictions
# binary_predictions = pd.DataFrame(binary_predictions, columns=y_test.columns)
y_test = y_test.astype(int)

print(predictions.head(15))
print(binary_predictions.head(15))
print(y_test.head(15))


test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)


predictable = []
not_predictable = []
ydrop_cols = []

for column in y_test.columns:
    corr = 1 - distance.jaccard(y_test[column], binary_predictions[column]) # jaccard of 0 is good, so we are doing 1 - jaccard to make it more intuitive
    print(f'{column}: {round(corr, 3)}')
    if corr > 0.5:
        predictable.append(column)
    else:
        ydrop_cols.append(column)
        not_predictable.append(column.split('_')[0])

print(f"These algorithms are reliably predictable: {predictable}")

# Removing noise (unpredictable algorithms) from training and test sets
Xdrop_cols = []
for col in X_train.columns:
    if col.split('_')[0] in not_predictable:
        Xdrop_cols.append(col)


X2_train = X_train.drop(labels=Xdrop_cols, axis=1)
X2_test = X_test.drop(labels=Xdrop_cols, axis=1)
y2_train = y_train.drop(labels=ydrop_cols, axis=1)
y2_test = y_test.drop(labels=ydrop_cols, axis=1)


X2_train_path = os.path.join(classification_data_dir, "x2_train.csv")
X2_test_path = os.path.join(classification_data_dir, "x2_test.csv")
y2_train_path = os.path.join(classification_data_dir, "y2_train.csv")
y2_test_path = os.path.join(classification_data_dir, "y2_test.csv")

X2_train.to_csv(X2_train_path, index=True, float_format="%.6f")
X2_test.to_csv(X2_test_path, index=True, float_format="%.6f")
y2_train.to_csv(y2_train_path, index=True)
y2_test.to_csv(y2_test_path, index=True)
