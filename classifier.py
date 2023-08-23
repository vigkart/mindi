import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

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

# Build the model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(90, activation="tanh"),
        tf.keras.layers.Dense(60, activation="elu"),
        tf.keras.layers.Dense(30, activation="elu"),
        tf.keras.layers.Dense(15, activation="relu"),
        tf.keras.layers.Dense(3, activation="sigmoid"),
    ]
)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=["accuracy"],
)

model.fit(X_train, y_train, epochs=100)

# Make the predictions
predictions = model.predict(X_test)
threshold = 0.5
binary_predictions = (predictions > threshold).astype(int)

predictions = pd.DataFrame(data=predictions, index=y_test.index)
binary_predictions = (predictions > 0.5).astype(float)
print(predictions.head(15))
print(binary_predictions.head(15))

# for i, pred in enumerate(predictions):
#     print(f'Sample {i+1}: {pred}')

# for i, pred in enumerate(binary_predictions):
#     print(f'Sample {i+1}: {pred}')


test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
