import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import joblib
import os

# Base directory
base_dir = os.path.dirname(__file__)

# Load dataset
data_path = os.path.join(base_dir, 'final_data.h5')
data = pd.read_hdf(data_path, key='df')

# Auto-select numeric columns
num_cols = [c for c in data.columns if data[c].dtype in ['float64', 'int64']]

# Pick top 3 numeric features for simplicity
features = num_cols[:3]
print(f"✅ Using features: {features}")

# Drop NaNs
data = data.dropna(subset=features)

# Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[features])

# Prepare X, y for time series (window = 1)
X, y = [], []
for i in range(1, len(scaled_data)):
    X.append(scaled_data[i - 1])
    y.append(scaled_data[i, 0])  # predict based on first column
X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], 1, X.shape[1]))  # LSTM expects [samples, timesteps, features]

# Build LSTM model
model = Sequential([
    LSTM(64, activation='tanh', input_shape=(1, len(features))),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=5, batch_size=32, verbose=1)

# Save model and scaler
model.save(os.path.join(base_dir, 'lstm_model.keras'))
joblib.dump(scaler, os.path.join(base_dir, 'scaler.pkl'))

print("🎯 Model and scaler rebuilt successfully!")
