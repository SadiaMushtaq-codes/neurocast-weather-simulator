import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import requests
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
# 1.  LOADING
@st.cache_resource
def load_assets():
    try:
        # Loading the .keras file
        model = load_model('weather_model.keras')
        with open('weather_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        return None, None
# 2. LIVE API DATA
@st.cache_data(ttl=600)
def get_live_data(_scaler):
    try:
        # Open-Meteo API for Hyderabad, Sindh
        api_url = (
            "https://api.open-meteo.com/v1/forecast?latitude=25.39&longitude=68.37"
            "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl"
            "&temperature_unit=fahrenheit&wind_speed_unit=kmh&precipitation_unit=inch"
            "&past_days=1&forecast_days=1"
        )
        response = requests.get(api_url)
        data = response.json()
        
        df = pd.DataFrame({
            'temp': data['hourly']['temperature_2m'],
            'humidity': data['hourly']['relative_humidity_2m'],
            'windspeed': data['hourly']['wind_speed_10m'],
            'sealevelpressure': data['hourly']['pressure_msl']
        })
        
        past_data = df.iloc[0:24]
        current = past_data.iloc[-1]
        current_input = (current['temp'], current['humidity'], current['windspeed'], current['sealevelpressure'])
        api_forecast = df['temp'].iloc[24:30].values
        
        # use _scaler here
        last_sequence_scaled = _scaler.transform(past_data)
        
        return current_input, last_sequence_scaled, api_forecast
    except:
        return None, None, None

# 3. FORECAST ENGINE
def get_ai_forecast(model, scaler, last_sequence, user_input, modifiers):
    t, h, w, p = user_input
    mt, mh, mw, mp = modifiers
    
    t, h, w, p = (t * mt, h * mh, w * mw, p * mp)
    
    user_scaled = scaler.transform([[t, h, w, p]])[0]
    seq = last_sequence.copy()
    seq = np.vstack([seq[1:], user_scaled])
    
    preds = []
    curr_seq = seq.copy()
    
    for _ in range(6):
        pr = model.predict(curr_seq.reshape(1, 24, 4), verbose=0)[0,0]
        preds.append(pr)
        new_row = curr_seq[-1].copy()
        new_row[0] = pr
        curr_seq = np.vstack([curr_seq[1:], new_row])
        
    dummy = np.zeros((6, 4))
    dummy[:, 0] = preds
    ai_temps = scaler.inverse_transform(dummy)[:,0]
    
    dummy_past = np.zeros((24, 4))
    dummy_past[:, 0] = last_sequence[:, 0]
    past_temps = scaler.inverse_transform(dummy_past)[:,0]
    past_temps = np.append(past_temps, t)
    
    return ai_temps, past_temps, (t, h, w, p)

# 4. UI
st.set_page_config(layout="wide", page_title="Hyderabad Weather AI")

# Load assets
assets = load_assets()
if assets[0] is None:
    st.error("😥 Model not found! Please run `python train_model.py` first.")
else:
    model, scaler = assets
    
    st.title("⛅NeuroCast: The Weather Simulator")
    st.markdown("University Project | **Live API Data + AI Benchmarking** |LSTM Weather AI Command Center")
    
    #Pass scaler to the function (it matches the _scaler argument)
    data_pack = get_live_data(scaler)
    
    if data_pack[0] is None:
        st.warning("Offline Mode: API unavailable. Using default values.")
        current_input = (85.0, 60.0, 12.0, 1005.0)
        last_seq = np.random.rand(24, 4)
        api_forecast = [0]*6
    else:
        current_input, last_seq, api_forecast = data_pack

    # Sidebar
    st.sidebar.header("Live Conditions (Hyderabad)")
    t = st.sidebar.slider("Temp (°F)", 40.0, 120.0, float(current_input[0]))
    h = st.sidebar.slider("Humidity (%)", 0.0, 100.0, float(current_input[1]))
    w = st.sidebar.slider("Wind (km/h)", 0.0, 50.0, float(current_input[2]))
    p = st.sidebar.slider("Pressure (hPa)", 980.0, 1040.0, float(current_input[3]))
    
    st.sidebar.divider()
    st.sidebar.header("⚡ Simulation")
    event = st.sidebar.selectbox("Event:", ["Normal", "Heatwave (+5%)", "Storm (Low Press)"])
    
    mods = {"Normal": (1,1,1,1), "Heatwave (+5%)": (1.05, 0.8, 1, 1), "Storm (Low Press)": (0.98, 1.2, 1.5, 0.98)}
    
    ai_forecast, past, sim_input = get_ai_forecast(model, scaler, last_seq, (t,h,w,p), mods[event])
    
    c1, c2 = st.columns([2,1])
    
    with c1:
        st.subheader("6-Hour AI Forecast")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(-24, 1), past, 'bo-', alpha=0.4, label='History (24h)')
        ax.plot(0, sim_input[0], 'go', markersize=10, label='Sim Start')
        ax.plot(range(1, 7), ai_forecast, 'rs-', linewidth=2, label='AI Model')
        
        if event == "Normal":
            ax.plot(range(1, 7), api_forecast, 'g--', alpha=0.7, label='Pro Benchmark')
            
        ax.set_ylabel("Temp (°F)")
        ax.set_xlabel("Hours from Now")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
    with c2:
        st.subheader("Simulation Stats")
        st.metric("Simulated Temp", f"{sim_input[0]:.1f} °F", delta=f"{sim_input[0]-t:.1f}")
        st.metric("Simulated Pressure", f"{sim_input[3]:.1f} hPa", delta=f"{sim_input[3]-p:.1f}")
        
        st.info("The graph compares your ML model (Red) against the professional Open-Meteo forecast (Green/Dashed).")

    st.divider()
    cols = st.columns(6)
    for i in range(6):
        cols[i].metric(f"+{i+1} Hour", f"{ai_forecast[i]:.1f} °F")