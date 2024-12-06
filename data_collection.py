import csv
import requests
from datetime import datetime


API_KEY = '9f86bd54629c353cb504d520e810030d'
CITY = "London"
URL = f"https://api.openweathermap.org/data/2.5/forecast?q={CITY}&appid={API_KEY}&units=metric"

def fetch_weather_data():
    response = requests.get(URL)
    response.raise_for_status()        # - Ensure the request was successful
    return response.json()["list"]

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

weather_data = fetch_weather_data()
save_weather_data(weather_data)
