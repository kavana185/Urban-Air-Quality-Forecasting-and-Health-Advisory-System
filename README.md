# Urban Air Quality Forecasting and Health Advisory System

A deep learning-based air quality forecasting system designed to predict short-term PM2.5 levels using pollution and meteorological data.

The system uses a CNN-LSTM architecture for time-series forecasting and provides health advisories based on predicted AQI levels through an interactive Streamlit dashboard.

---

## Features

- Short-term PM2.5 forecasting using CNN-LSTM
- Multi-step AQI prediction (1–6 hours ahead)
- Health advisory generation based on pollution severity
- Time-series preprocessing and feature engineering
- Interactive Streamlit dashboard
- Real-time visualization of AQI trends and forecasts

---

## Tech Stack

| Category | Technologies |
|---|---|
| Machine Learning | TensorFlow/Keras, Scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Streamlit |
| Deployment | Streamlit, ngrok |
| Language | Python |

---
## Dataset

The model was trained using the **Air Quality Dataset: Indian Cities (2022–2025)** from Kaggle.

Dataset Highlights:
- 842,000+ hourly observations
- 29 major Indian cities
- 63 environmental and meteorological features
- Includes pollution, weather, and temporal data

Dataset Link:  
https://www.kaggle.com/datasets/bhautikvekariya21/air-quality-dataset-indian-cities-2022-2025

## Input Features

### Air Pollutants
- PM2.5
- PM10
- NO₂
- CO
- O₃

### Meteorological Features
- Temperature
- Humidity
- Wind Speed
- Pressure

### Temporal Features
- Hour
- Day
- Month

---

## Model Workflow

1. Data Collection and Cleaning
2. Missing Value Handling
3. Feature Engineering
4. Time-Series Sequence Generation
5. CNN-LSTM Training
6. Multi-step AQI Forecasting
7. Health Advisory Generation
8. Dashboard Visualization

---

## Forecasting Objective

Predict short-term PM2.5 concentration levels for:
- Next 1 hour
- Up to next 6 hours recursively

---

## Health Advisory System

The system generates AQI-based health advisories for:
- General Public
- Children
- Elderly Individuals
- People with Respiratory Conditions

---

## Model Highlights

- CNN layers extract local temporal patterns
- LSTM layers capture long-term dependencies
- Sliding window sequence generation using 24 time steps
- Multi-step recursive forecasting approach

---

## Live Demo

https://wrist-ipad-scotch.ngrok-free.dev/

---

## Repository

https://github.com/kavana185/Urban-Air-Quality-Forecasting-and-Health-Advisory-System

---

## Installation

```bash
git clone https://github.com/kavana185/Urban-Air-Quality-Forecasting-and-Health-Advisory-System
cd Urban-Air-Quality-Forecasting-and-Health-Advisory-System
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## Future Improvements

- City-wise AQI forecasting
- Satellite data integration
- Transformer-based forecasting models
- Real-time sensor integration
- Mobile application deployment
