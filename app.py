from flask import Flask, render_template, request
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import logging
import random
import joblib
from sklearn.linear_model import LinearRegression
import requests
import os

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

CSV_FILE = 'building_data.csv'
API_KEY = "cbf087aa603ca865dcb547f48f3f72ed"  # Replace with your OpenWeatherMap API key
CITY = "Bhilai"         # Replace with your city name

# Load or train machine learning model
def load_or_train_model(df):
    try:
        model = joblib.load('energy_model.pkl')
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.info("Training new model.")
        X = df[['Temperature', 'Humidity', 'Occupancy']]
        y = df['EnergyConsumption']
        model = LinearRegression().fit(X, y)
        joblib.dump(model, 'energy_model.pkl')
        logging.info("Model trained and saved.")
    return model

# Function to fetch real weather data from OpenWeatherMap API
def fetch_weather_data():
    url = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        logging.error("Failed to fetch weather data: " + response.text)
        return None
    data = response.json()
    try:
        # Extract temperature and humidity from the API response
        temp = data['main']['temp']
        humidity = data['main']['humidity']
    except KeyError as e:
        logging.error("Invalid response structure: missing key " + str(e))
        return None
    
    # For values not provided by the weather API, we simulate them
    occupancy = random.randint(0, 1)
    # Use the current model (if available) to predict energy consumption from current weather and occupancy.
    try:
        energy_consumption = model.predict([[temp, humidity, occupancy]])[0]
    except Exception as e:
        logging.error("Model prediction failed: " + str(e))
        energy_consumption = random.uniform(10, 50)
    
    new_data = {
        'Timestamp': datetime.now(),
        'Temperature': temp,
        'Humidity': humidity,
        'Occupancy': occupancy,
        'EnergyConsumption': energy_consumption
    }
    return new_data

# Initialize or load CSV data
def load_csv_data():
    if os.path.isfile(CSV_FILE):
        df = pd.read_csv(CSV_FILE, parse_dates=['Timestamp'])
    else:
        # If CSV does not exist, fetch current weather data to initialize it
        new_row = fetch_weather_data()
        if new_row is None:
            # Fallback in case API fails: generate sample data
            new_row = {
                'Timestamp': datetime.now(),
                'Temperature': 25.0,
                'Humidity': 50.0,
                'Occupancy': random.randint(0, 1),
                'EnergyConsumption': random.uniform(10, 50)
            }
        df = pd.DataFrame([new_row])
        df.to_csv(CSV_FILE, index=False)
    return df

# Global DataFrame loaded from CSV
df = load_csv_data()
# Load or train the model using the current DataFrame
model = load_or_train_model(df)

# Predictive analytics
def predict_conditions(df, model):
    try:
        df = df.set_index('Timestamp')
        future_times = pd.date_range(start=df.index[-1], periods=25, freq='H')[1:]
        temperature_pred = df['Temperature'].rolling(window=3).mean().iloc[-1] + np.random.uniform(-1, 1, size=24)
        humidity_pred = df['Humidity'].rolling(window=3).mean().iloc[-1] + np.random.uniform(-1, 1, size=24)
        occupancy_pred = np.random.randint(0, 2, size=24)

        X_future = pd.DataFrame({
            'Temperature': temperature_pred,
            'Humidity': humidity_pred,
            'Occupancy': occupancy_pred
        })
        energy_pred = model.predict(X_future)

        future_df = pd.DataFrame({
            'Timestamp': future_times,
            'Temperature': temperature_pred,
            'Humidity': humidity_pred,
            'Occupancy': occupancy_pred,
            'EnergyConsumption': energy_pred
        })
        return future_df
    except Exception as e:
        logging.error(f"Error in predictive analytics: {e}")
        return pd.DataFrame()

@app.route('/', methods=['GET', 'POST'])
def index():
    global df, model
    custom_plot = request.form.get('plot_type', 'Temperature and Humidity')
    
    # If the form is submitted (i.e. user clicked Update), fetch real weather data
    if request.method == 'POST':
        new_row = fetch_weather_data()
        if new_row:
            df_new = pd.DataFrame([new_row])
            # Append new row to CSV file if it exists; otherwise, create a new CSV
            if os.path.isfile(CSV_FILE):
                df_new.to_csv(CSV_FILE, mode='a', header=False, index=False)
            else:
                df_new.to_csv(CSV_FILE, index=False)
            logging.info("Weather data updated via OpenWeatherMap API.")
            # Reload the global DataFrame from CSV
            df = pd.read_csv(CSV_FILE, parse_dates=['Timestamp'])
        else:
            logging.error("Failed to update weather data. Using existing data.")

    # Compute predictive analytics based on the current CSV data
    future_df = predict_conditions(df, model)

    if custom_plot == 'Energy Consumption':
        fig = px.line(df, x='Timestamp', y='EnergyConsumption', title='Energy Consumption')
        fig.update_xaxes(title='Time')
        fig.update_yaxes(title='Energy Consumption (kWh)')
        fig.add_trace(go.Scatter(x=future_df['Timestamp'], y=future_df['EnergyConsumption'], mode='lines', name='Predicted Energy Consumption'))
    else:
        fig = px.line(df, x='Timestamp', y=['Temperature', 'Humidity'], title='Building Conditions')
        fig.update_xaxes(title='Time')
        fig.update_yaxes(title='Value')
        fig.add_trace(go.Scatter(x=future_df['Timestamp'], y=future_df['Temperature'], mode='lines', name='Predicted Temperature'))
        fig.add_trace(go.Scatter(x=future_df['Timestamp'], y=future_df['Humidity'], mode='lines', name='Predicted Humidity'))

    plot_div = fig.to_html(full_html=False, include_plotlyjs='cdn')

    return render_template('index.html', plot_div=plot_div, custom_plot=custom_plot)

if __name__ == '__main__':
    app.run(debug=True)
