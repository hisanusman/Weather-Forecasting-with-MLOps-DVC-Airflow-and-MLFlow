# Import necessary libraries
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import os
import numpy as np
import pandas as pd
import pickle

# Set the local storage path
LOCAL_STORAGE_PATH = r'E:\\FAST_NUCES\\SEMISTER_7_BS(AI)\\MLOPS\\project\\MLOps-Activity7\\MLOps-Activity7'

# Preprocess Data
def preprocess_data(input_file, output_file):
    df = pd.read_csv(input_file)

    # Normalize numerical columns
    numerical_columns = ['Temperature', 'Humidity', 'Wind Speed']
    means = df[numerical_columns].mean(axis=0)
    stds = df[numerical_columns].std(axis=0)
    df[numerical_columns] = (df[numerical_columns] - means) / stds

    # Encode categorical data
    conditions = df['Weather Condition'].unique()
    condition_to_label = {condition: idx for idx, condition in enumerate(conditions)}
    df['condition_encoded'] = df['Weather Condition'].map(condition_to_label)

    # Save processed data
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to: {output_file}")


# Train Model with MLflow integration
def train_model(input_file, model_file):
    # Load processed data
    df = pd.read_csv(input_file)

    # Prepare features and target
    X = df[["Humidity", "Wind Speed", "condition_encoded"]].values
    y = df["Temperature"].values

    # Train/test split
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    test_size = int(0.2 * len(X))
    train_indices = indices[test_size:]
    test_indices = indices[:test_size]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    # Train a linear regression model using the Normal Equation
    X_train_bias = np.c_[np.ones(X_train.shape[0]), X_train]
    X_test_bias = np.c_[np.ones(X_test.shape[0]), X_test]
    theta = np.linalg.inv(X_train_bias.T @ X_train_bias) @ X_train_bias.T @ y_train

    # Save the trained model
    model = {"coefficients": theta[1:], "intercept": theta[0]}
    with open(model_file, "wb") as f:
        pickle.dump(model, f)
    print(f"Trained model saved to: {model_file}")

    # Start MLflow run
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("Weather Data Pipeline")

    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params({"test_size": 0.2, "random_seed": 42})

        # Log metrics (example: train and test accuracy)
        y_train_pred = X_train_bias @ theta
        y_test_pred = X_test_bias @ theta
        train_accuracy = 1 - np.mean((y_train - y_train_pred) ** 2) / np.var(y_train)
        test_accuracy = 1 - np.mean((y_test - y_test_pred) ** 2) / np.var(y_test)
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)

        # Set a tag
        mlflow.set_tag("Model Type", "Linear Regression")

        # Infer the model signature
        signature = infer_signature(X_train, y_train_pred)

        # Log the model
        mlflow.sklearn.log_model(
            sk_model=None,  # Placeholder for custom model
            artifact_path="weather_model",
            registered_model_name="weather_forecast_model",
            signature=signature,
        )

        print("Model logged to MLflow.")

    # Track with DVC
    os.system(f"dvc add {model_file}")
    os.system(f"git add {model_file}.dvc")
    os.system("git commit -m 'Add trained model'")
    os.system("dvc push")


# Main workflow
if __name__ == "__main__":
    raw_data_file = os.path.join(LOCAL_STORAGE_PATH, "raw_data.csv")
    processed_data_file = os.path.join(LOCAL_STORAGE_PATH, "processed_weather_data.csv")
    model_file_path = os.path.join(LOCAL_STORAGE_PATH, "model.pkl")

    # Step 1: Preprocess data
    preprocess_data(raw_data_file, processed_data_file)

    # Step 2: Train model
    train_model(processed_data_file, model_file_path)
