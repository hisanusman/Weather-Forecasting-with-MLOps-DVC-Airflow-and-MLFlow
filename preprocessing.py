import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_weather_data(input_file="raw_data.csv", output_file="processed_data.csv"):
    # Load the data
    data = pd.read_csv(input_file)

    # Handle missing values
    data.fillna(data.mean(), inplace=True)

    # Encode categorical values (e.g., "Weather Condition")
    if "Weather Condition" in data.columns:
        label_encoder = LabelEncoder()
        data["Weather Condition"] = label_encoder.fit_transform(data["Weather Condition"])

    # Normalize numerical fields
    numerical_features = ["Temperature", "Humidity", "Wind Speed"]
    data[numerical_features] = (data[numerical_features] - data[numerical_features].mean()) / data[numerical_features].std()

    # Save the processed data
    data.to_csv(output_file, index=False)

    print("Preprocessing complete. Processed data saved to:", output_file)

preprocess_weather_data()
