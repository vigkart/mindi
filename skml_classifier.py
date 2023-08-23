from skmultilearn.problem_transform import BinaryRelevance
from sklearn.linear_model import SGDClassifier
from scipy.sparse import lil_matrix
from sklearn.svm import SVC
import os
import pandas as pd

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

# # Converting y_train & y_test to sparse matrices
# y_train = lil_matrix(y_train.values)
# y_test = lil_matrix(y_test.values)

# clf_base = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3)

# classifier = BinaryRelevance(classifier=clf_base, require_dense=[False, True])
# classifier.fit(X_train, y_train)

# y_pred = classifier.predict(X_test)
# print(y_pred.toarray())

print(X_train)
