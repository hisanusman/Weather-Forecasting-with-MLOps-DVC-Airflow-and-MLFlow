import pandas as pd
import numpy as np
import pickle

def train_model(input_file="processed_data.csv", model_file="model.pkl"):
    data = pd.read_csv(input_file)

    X = data[["Humidity", "Wind Speed"]].values
    y = data["Temperature"].values  

    np.random.seed(42)
    indices = np.random.permutation(len(X))
    test_size = int(0.2 * len(X))
    train_indices = indices[test_size:]
    test_indices = indices[:test_size]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    X_train_bias = np.c_[np.ones(X_train.shape[0]), X_train]
    X_test_bias = np.c_[np.ones(X_test.shape[0]), X_test]

    theta = np.linalg.inv(X_train_bias.T @ X_train_bias) @ X_train_bias.T @ y_train

    y_pred = X_test_bias @ theta


    model = {"coefficients": theta[1:], "intercept": theta[0]}
    with open(model_file, "wb") as file:
        pickle.dump(model, file)

train_model()
