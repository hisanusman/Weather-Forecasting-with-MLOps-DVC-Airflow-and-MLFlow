import os
import datetime
import pandas as pd
import numpy as np
import requests
import pickle
from airflow import DAG
from airflow.operators.python import PythonOperator

# - Define the local storage path dynamically based on the current working directory
LOCAL_STORAGE_PATH = os.getcwd()
DVC_REMOTE = os.path.join(LOCAL_STORAGE_PATH, "dvc-storage")

# - Define the API URL and key
API_URL = "http://api.openweathermap.org/data/2.5/forecast"
API_KEY = "API_KEY"  # Replace with your API key

# -- Fetch Weather Data
def fetch_weather_data():
    params = {
        "q": "London,GB",   
        "appid": API_KEY,
        "units": "metric",  
        "cnt": 100,         
    }
    response = requests.get(API_URL, params=params)
    response.raise_for_status()  # Raise an exception for HTTP errors
    data = response.json()

    # - Extract relevant fields
    weather_data = []
    for entry in data["list"]:
        weather_data.append({
            "date_time": entry["dt_txt"],
            "temperature": entry["main"]["temp"],
            "humidity": entry["main"]["humidity"],
            "wind_speed": entry["wind"]["speed"],
            "condition": entry["weather"][0]["description"],
        })

    # - Save the raw data
    raw_file_path = os.path.join(LOCAL_STORAGE_PATH, "raw_weather_data.csv")
    df = pd.DataFrame(weather_data)
    df.to_csv(raw_file_path, index=False)
    print(f"Raw weather data saved to: {raw_file_path}")

    # - Track with DVC
    os.system(f"dvc add {raw_file_path}")
    os.system(f"git add {raw_file_path}.dvc")
    os.system("git commit -m 'Add raw weather data'")
    os.system("dvc push")

# -- Preprocess Data
def preprocess_data():
    raw_file_path = os.path.join(LOCAL_STORAGE_PATH, "raw_weather_data.csv")
    processed_file_path = os.path.join(LOCAL_STORAGE_PATH, "processed_weather_data.csv")

    # - Check if the raw file exists before proceeding
    if not os.path.exists(raw_file_path):
        raise FileNotFoundError(f"{raw_file_path} not found. Please ensure data fetch was successful.")

    # - Load raw data
    df = pd.read_csv(raw_file_path)

    # - Fill missing values for numerical columns with mean and categorical with mode
    df["temperature"].fillna(df["temperature"].mean(), inplace=True)
    df["humidity"].fillna(df["humidity"].mean(), inplace=True)
    df["wind_speed"].fillna(df["wind_speed"].mean(), inplace=True)
    df["condition"].fillna(df["condition"].mode()[0], inplace=True)

    # - Standardize numerical data
    numerical_columns = ['temperature', 'humidity', 'wind_speed']
    df[numerical_columns] = (df[numerical_columns] - df[numerical_columns].mean()) / df[numerical_columns].std()

    # - Encode categorical data (Weather Condition)
    df["condition_encoded"] = df["condition"].astype('category').cat.codes

    # - Save processed data
    df.to_csv(processed_file_path, index=False)
    print(f"Processed weather data saved to: {processed_file_path}")

    # - Track with DVC
    os.system(f"dvc add {processed_file_path}")
    os.system(f"git add {processed_file_path}.dvc")
    os.system("git commit -m 'Add processed weather data'")
    os.system("dvc push")

# -- Train Model
def train_model():
    processed_file_path = os.path.join(LOCAL_STORAGE_PATH, "processed_weather_data.csv")
    model_file_path = os.path.join(LOCAL_STORAGE_PATH, "model.pkl")

    # - Check if the processed file exists
    if not os.path.exists(processed_file_path):
        raise FileNotFoundError(f"{processed_file_path} not found. Please ensure data preprocessing was successful.")

    # - Load processed data
    df = pd.read_csv(processed_file_path)

    # - Prepare features and target
    X = df[["humidity", "wind_speed", "condition_encoded"]].values
    y = df["temperature"].values

    # - Train/test split
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    test_size = int(0.2 * len(X))
    train_indices = indices[test_size:]
    test_indices = indices[:test_size]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    # - Train a linear regression model using the Normal Equation
    X_train_bias = np.c_[np.ones(X_train.shape[0]), X_train]
    X_test_bias = np.c_[np.ones(X_test.shape[0]), X_test]
    theta = np.linalg.inv(X_train_bias.T @ X_train_bias) @ X_train_bias.T @ y_train

    # - Save the trained model
    model = {"coefficients": theta[1:], "intercept": theta[0]}
    with open(model_file_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Trained model saved to: {model_file_path}")

    # - Track with DVC
    os.system(f"dvc add {model_file_path}")
    os.system(f"git add {model_file_path}.dvc")
    os.system("git commit -m 'Add trained model'")
    os.system("dvc push")

# - Define the DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime.datetime(2024, 11, 26),
    "retries": 1,
}
dag = DAG(
    "weather_data_pipeline",
    default_args=default_args,
    description="A pipeline to fetch, preprocess weather data, and train a model",
    schedule_interval=datetime.timedelta(days=1),
)

# - Define tasks
fetch_data_task = PythonOperator(
    task_id="fetch_weather_data",
    python_callable=fetch_weather_data,
    dag=dag,
)

preprocess_data_task = PythonOperator(
    task_id="preprocess_data",
    python_callable=preprocess_data,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id="train_model",
    python_callable=train_model,
    dag=dag,
)




# - Task dependencies
fetch_data_task >> preprocess_data_task >> train_model_task
