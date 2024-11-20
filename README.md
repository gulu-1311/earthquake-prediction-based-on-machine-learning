# earthquake-prediction-based-on-machine-learning
#prediction  the earthquake by using past data of earthquake  
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load historical earthquake data
def load_historical_data(filepath):
    data = pd.read_csv(filepath)
    return data

# Train a machine learning model
def train_model(data):
    X = data[['latitude', 'longitude', 'depth']]
    y = data['magnitude']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Save the trained model
def save_model(model, filepath):
    joblib.dump(model, filepath)

# Load a trained model
def load_model(filepath):
    return joblib.load(filepath)

# Function to fetch data from the USGS API
def fetch_data(latitude, longitude, depth, start_date, end_date):
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        'format': 'geojson',
        'starttime': start_date,
        'endtime': end_date,
        'minmagnitude': '4.0',
        'latitude': str(latitude),
        'longitude': str(longitude),
        'maxradiuskm': '500',
        'orderby': 'time'
    }
    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"Error: Received status code {response.status_code}")
        print("Response content:", response.text)
        return None

    try:
        data = response.json()
    except requests.exceptions.JSONDecodeError:
        print("Error: Failed to decode JSON from response")
        print("Response content:", response.text)
        return None

    return data

# Function to evaluate recent past seismic activity
def recent_earthquake_analysis(data):
    if not data or 'features' not in data or not data['features']:
        return "No recent earthquakes found in the vicinity."

    recent_earthquakes = data['features']
    recent_mags = [quake['properties']['mag'] for quake in recent_earthquakes]

    average_mag = np.mean(recent_mags)
    if average_mag > 6.0:
        return "High seismic hazard detected."
    elif average_mag > 4.5:
        return "Moderate seismic hazard detected."
    else:
        return "Low seismic hazard."

# Function to plot seismic activity over the past five years
def plot_seismic_activity(data):
    if not data or 'features' not in data or not data['features']:
        print("No data available to plot.")
        return

    dates = [datetime.utcfromtimestamp(quake['properties']['time'] / 1000) for quake in data['features']]
    magnitudes = [quake['properties']['mag'] for quake in data['features']]

    plt.figure(figsize=(10, 5))
    plt.plot(dates, magnitudes, 'o-', markersize=4)
    plt.xlabel('Date')
    plt.ylabel('Magnitude')
    plt.title('Seismic Activity Over the Past Five Years')
    plt.grid(True)
    plt.show()

# Function to predict seismic hazard using the ML model
def predict_seismic_hazard(model, latitude, longitude, depth):
    input_data = np.array([[latitude, longitude, depth]])
    predicted_magnitude = model.predict(input_data)[0]
    print(f"Predicted Magnitude: {predicted_magnitude:.2f}")

    if predicted_magnitude > 6.0:
        hazard_level = "High seismic hazard detected."
    elif predicted_magnitude > 4.5:
        hazard_level = "Moderate seismic hazard detected."
    else:
        hazard_level = "Low seismic hazard."

    # Fetch and plot seismic activity data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    data = fetch_data(latitude, longitude, depth, start_date, end_date)
    if data:
        print("Plotting seismic activity...")
        plot_seismic_activity(data)

    return hazard_level

# Main program loop to accept user input and make predictions
if _name_ == "_main_":
    # Load historical data and train model
    historical_data = load_historical_data('/content/earthquake(2024-2020).csv')
    model = train_model(historical_data)
    save_model(model, 'seismic_model.pkl')
    model = load_model('seismic_model.pkl')

    while True:
        try:
            latitude = float(input("Enter latitude: "))
            longitude = float(input("Enter longitude: "))
            depth = float(input("Enter height from sea level: "))

            print("Predicting seismic hazard...")
            hazard_prediction = predict_seismic_hazard(model, latitude, longitude, depth)
            print(hazard_prediction)
        except Exception as e:
            print(f"An error occurred: {e}")

        cont = input("Do you want to enter another location? (yes/no): ")
        if cont.lower() != 'yes':
            break
