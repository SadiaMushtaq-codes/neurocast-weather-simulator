import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

print("1. Libraries imported.")

# 2. FETCH HIGH-QUALITY HISTORICAL DATA (API)
# Hyderabad, Sindh coordinates
latitude = 25.39
longitude = 68.37

# Fetching Feb, March, April 2024 data
api_url = (
    f"https://archive-api.open-meteo.com/v1/archive?latitude={latitude}&longitude={longitude}"
    "&start_date=2024-02-01&end_date=2024-04-30"
    "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl"
    "&temperature_unit=fahrenheit&wind_speed_unit=kmh&precipitation_unit=inch"
)

print("2. Fetching training data from Open-Meteo...")
response = requests.get(api_url)
data = response.json()

hourly_data = data['hourly']
df = pd.DataFrame({
    'datetime': pd.to_datetime(hourly_data['time']),
    'temp': hourly_data['temperature_2m'],
    'humidity': hourly_data['relative_humidity_2m'],
    'windspeed': hourly_data['wind_speed_10m'],
    'sealevelpressure': hourly_data['pressure_msl']
})

df = df.set_index('datetime').dropna()
print(f"   Loaded {len(df)} rows of high-quality training data.")

# 3. SCALE AND SEQUENCE
print("3. Processing data (Scaling and Sequencing)...")

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df)

SEQ_LEN = 24
X, y = [], []
for i in range(len(data_scaled)-SEQ_LEN):
    X.append(data_scaled[i:i+SEQ_LEN])
    y.append(data_scaled[i+SEQ_LEN, 0]) # Predict temp

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

split = int(0.8*len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"   Training samples: {X_train.shape[0]}")
# 4. TRAIN MODEL
print("4. Building and Training LSTM Model...")

model = Sequential([
    LSTM(50, input_shape=(SEQ_LEN, 4)),
    Dense(1)])
model.compile(optimizer=Adam(0.001), loss='mse')

print("   Starting training (this may take a moment)...")
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)
print("   Training complete.")


print("5. Saving model assets...")

# Saving as .keras to fix the warning
model.save('weather_model.keras')

with open('weather_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("SUCCESS: Assets saved (weather_model.keras, weather_scaler.pkl)")