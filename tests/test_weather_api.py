import pytest
import requests
from unittest import mock
import csv
from datetime import datetime
import pandas as pd

# API Configuration
API_KEY = '9f86bd54629c353cb504d520e810030d'
CITY = "London"
URL = f"https://api.openweathermap.org/data/2.5/forecast?q={CITY}&appid={API_KEY}&units=metric"

# Function to fetch weather data
def fetch_weather_data():
    response = requests.get(URL)
    response.raise_for_status()  # Ensure the request was successful
    return response.json()["list"]

# Function to save weather data to a CSV file
def save_weather_data(data, filename="raw_data.csv"):
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Date", "Time", "Temperature", "Humidity", "Wind Speed", "Weather Condition"])
        for entry in data:
            date_time = datetime.fromtimestamp(entry["dt"]).strftime("%Y-%m-%d %H:%M:%S")
            date, time = date_time.split(" ")
            writer.writerow([
                date,
                time,
                entry["main"]["temp"],
                entry["main"]["humidity"],
                entry["wind"]["speed"],
                entry["weather"][0]["description"]
            ])

# Mock data to simulate the API response
mock_weather_data = {
    "list": [
        {
            "dt": 1633036800,  # Timestamp for a specific date and time
            "main": {
                "temp": 22.5,
                "humidity": 60
            },
            "wind": {
                "speed": 5
            },
            "weather": [
                {
                    "description": "Clear sky"
                }
            ]
        }
    ]
}

# Test fetch_weather_data
@mock.patch('requests.get')
def test_fetch_weather_data(mock_get):
    # Mock the response object
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = mock_weather_data

    # Call the function
    weather_data = fetch_weather_data()

    # Assert that the correct data is returned
    assert len(weather_data) == 1
    assert weather_data[0]["main"]["temp"] == 22.5
    assert weather_data[0]["weather"][0]["description"] == "Clear sky"

# Test save_weather_data
@mock.patch('builtins.open', new_callable=mock.mock_open)
@mock.patch('csv.writer')
def test_save_weather_data(mock_writer, mock_file):
    # Mock the writer object
    mock_csv_writer = mock_writer.return_value

    # Call the function
    save_weather_data(mock_weather_data['list'], filename="test_weather.csv")

    # Assert that the file was opened correctly
    mock_file.assert_called_once_with('test_weather.csv', 'w', newline='')





    # Assert that the correct rows were written
    mock_csv_writer.writerow.assert_any_call(["Date", "Time", "Temperature", "Humidity", "Wind Speed", "Weather Condition"])
    from datetime import datetime

    expected_date_time = datetime.fromtimestamp(1633036800).strftime("%Y-%m-%d %H:%M:%S")
    expected_date, expected_time = expected_date_time.split(" ")

    mock_csv_writer.writerow.assert_any_call([
        expected_date,  # Dynamically calculated date
        expected_time,  # Dynamically calculated time
        22.5,
        60,
        5,
        "Clear sky"
    ])
