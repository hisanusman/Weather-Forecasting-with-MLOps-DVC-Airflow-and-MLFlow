import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

def train_model(input_file="processed_data.csv", model_file="model.pkl"):
    # Load processed data
    data = pd.read_csv(input_file)

    # Prepare features and target
    X = data[["Humidity", "Wind Speed"]]
    y = data["Temperature"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the model
    with open(model_file, "wb") as file:
        pickle.dump(model, file)

train_model()