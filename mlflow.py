import os
import datetime
import pandas as pd
import numpy as np
import requests
import mlflow
import mlflow.sklearn

# Define experiment and tracking URI
mlflow.set_experiment("weather_data_pipeline")

# - Define paths and API configurations
LOCAL_STORAGE_PATH = os.getcwd()
API_URL = "http://api.openweathermap.org/data/2.5/forecast"
API_KEY = "API_KEY"  # Replace with your API key

def fetch_weather_data():
    params = {"q": "London,GB", "appid": API_KEY, "units": "metric", "cnt": 100}
    response = requests.get(API_URL, params=params)
    response.raise_for_status()
    data = response.json()
    
    weather_data = []
    for entry in data["list"]:
        weather_data.append({
            "date_time": entry["dt_txt"],
            "temperature": entry["main"]["temp"],
            "humidity": entry["main"]["humidity"],
            "wind_speed": entry["wind"]["speed"],
            "condition": entry["weather"][0]["description"],
        })
    
    df = pd.DataFrame(weather_data)
    raw_file_path = os.path.join(LOCAL_STORAGE_PATH, "raw_weather_data.csv")
    df.to_csv(raw_file_path, index=False)
    mlflow.log_artifact(raw_file_path)

def preprocess_data():
    raw_file_path = os.path.join(LOCAL_STORAGE_PATH, "raw_weather_data.csv")
    processed_file_path = os.path.join(LOCAL_STORAGE_PATH, "processed_weather_data.csv")

    # Load and preprocess data
    df = pd.read_csv(raw_file_path)
    df["temperature"].fillna(df["temperature"].mean(), inplace=True)
    df["humidity"].fillna(df["humidity"].mean(), inplace=True)
    df["wind_speed"].fillna(df["wind_speed"].mean(), inplace=True)
    df["condition"].fillna(df["condition"].mode()[0], inplace=True)
    
    numerical_columns = ['temperature', 'humidity', 'wind_speed']
    df[numerical_columns] = (df[numerical_columns] - df[numerical_columns].mean()) / df[numerical_columns].std()
    
    df["condition_encoded"] = df["condition"].astype('category').cat.codes
    df.to_csv(processed_file_path, index=False)

    mlflow.log_artifact(processed_file_path)

def train_model():
    processed_file_path = os.path.join(LOCAL_STORAGE_PATH, "processed_weather_data.csv")
    df = pd.read_csv(processed_file_path)

    X = df[["humidity", "wind_speed", "condition_encoded"]].values
    y = df["temperature"].values

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

    # Log model
    model = {"coefficients": theta[1:], "intercept": theta[0]}
    
    with mlflow.start_run():
        mlflow.log_param("model_type", "linear_regression")
        mlflow.pyfunc.log_model("weather_model", python_model=model)

if __name__ == "__main__":
    fetch_weather_data()
    preprocess_data()
    train_model()
