import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib

# ------------------ LOAD DATA ------------------
df = pd.read_csv("INDIA_AQI_COMPLETE_20251126.csv")

df = df.ffill().bfill()
df['Datetime'] = pd.to_datetime(df['Datetime'])

df['hour'] = df['Datetime'].dt.hour
df['day'] = df['Datetime'].dt.day
df['month'] = df['Datetime'].dt.month

features = [
    'PM2_5_ugm3', 'PM10_ugm3', 'NO2_ugm3', 'CO_ugm3', 'O3_ugm3',
    'Temp_2m_C', 'Humidity_Percent', 'Wind_Speed_10m_kmh',
    'Pressure_MSL_hPa', 'hour', 'day', 'month'
]

df = df[features]

# ------------------ LOAD SCALER ------------------
scaler = joblib.load("scaler.pkl")
scaled_data = scaler.transform(df)

# ------------------ LOAD MODEL ------------------
model = tf.keras.models.load_model("aqi_model (2).keras", compile=False)

# ------------------ UI ------------------
st.set_page_config(page_title="Air Quality Forecast", layout="wide")

st.title("🌍 Urban Air Quality Forecasting System")

st.sidebar.header("Simulation Settings")
st.sidebar.write("Using latest 24 hours data")

# ------------------ INPUT DATA ------------------
input_df = df.iloc[-25:-1]
actual_next = df.iloc[-1]['PM2_5_ugm3']

scaled_input = scaled_data[-25:-1]
input_data = np.expand_dims(scaled_input, axis=0)

# ------------------ PREDICTION ------------------
prediction = model.predict(input_data, verbose=0)

dummy = np.zeros((1, len(features)))
dummy[0, 0] = prediction[0][0]

pred_pm25 = scaler.inverse_transform(dummy)[0][0]

# ------------------ MULTI-STEP FORECAST ------------------
future_preds = []
temp_input = scaled_input.copy()

for _ in range(6):  # next 6 hours
    inp = np.expand_dims(temp_input, axis=0)
    pred = model.predict(inp, verbose=0)[0][0]
    future_preds.append(pred)

    new_row = temp_input[-1].copy()
    new_row[0] = pred
    temp_input = np.vstack([temp_input[1:], new_row])

future_pm25 = []
for p in future_preds:
    dummy = np.zeros((1, len(features)))
    dummy[0, 0] = p
    val = scaler.inverse_transform(dummy)[0][0]
    future_pm25.append(val)

# ------------------ ERROR ------------------
error = abs(pred_pm25 - actual_next)

# ------------------ GROUP ADVISORY ------------------
def get_advisory(pm25):
    if pm25 <= 30:
        return {
            "General": "Safe for all activities",
            "Children": "Safe to play outdoors",
            "Elderly": "No restrictions",
            "Respiratory": "No risk"
        }
    elif pm25 <= 60:
        return {
            "General": "Moderate air quality",
            "Children": "Limit prolonged outdoor play",
            "Elderly": "Avoid long exposure",
            "Respiratory": "Be cautious"
        }
    elif pm25 <= 100:
        return {
            "General": "Poor air quality",
            "Children": "Avoid outdoor activity",
            "Elderly": "Stay indoors",
            "Respiratory": "Use masks"
        }
    else:
        return {
            "General": "Hazardous air quality",
            "Children": "Stay indoors",
            "Elderly": "Strictly avoid outdoors",
            "Respiratory": "Use purifiers and masks"
        }

advisory = get_advisory(pred_pm25)

# ------------------ DISPLAY ------------------
col1, col2 = st.columns(2)

with col1:
    st.metric("Predicted PM2.5", f"{pred_pm25:.2f}")
    st.metric("Actual PM2.5", f"{actual_next:.2f}")
    st.metric("Absolute Error", f"{error:.2f}")

    st.write("### Advisory by Group")
    for group, advice in advisory.items():
        st.write(f"**{group}:** {advice}")

    st.write("### Input Summary")
    st.write(f"Avg: {input_df['PM2_5_ugm3'].mean():.2f}")
    st.write(f"Max: {input_df['PM2_5_ugm3'].max():.2f}")
    st.write(f"Min: {input_df['PM2_5_ugm3'].min():.2f}")

with col2:
    st.write("### Actual vs Predicted")

    actual = input_df['PM2_5_ugm3'].values

    fig, ax = plt.subplots()

    ax.plot(range(24), actual, label="Input", marker='o')
    ax.plot(24, pred_pm25, 'ro', label="Predicted")
    ax.plot(24, actual_next, 'go', label="Actual")

    ax.legend()
    st.pyplot(fig)

# ------------------ FUTURE FORECAST ------------------
st.write("### Future Forecast (Next 6 Hours)")

fig3, ax3 = plt.subplots()
ax3.plot(range(24), actual, label="Input")
ax3.plot(range(24, 30), future_pm25, 'ro--', label="Forecast")

ax3.legend()
st.pyplot(fig3)

st.markdown("---")
st.caption("ML-powered air quality forecasting system 🚀")