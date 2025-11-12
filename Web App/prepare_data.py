import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ---------------------------------------------------------------------
# ✅ Load dataset
# ---------------------------------------------------------------------
data_path = os.path.join(os.path.dirname(__file__), "final_data.h5")
if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ File not found: {data_path}")

print(f"📂 Loading dataset from {data_path} ...")
data = pd.read_hdf(data_path, key="df")

print(f"✅ Data loaded successfully: {data.shape} rows and columns\n")

# ---------------------------------------------------------------------
# ✅ Select relevant columns for training
# ---------------------------------------------------------------------
# Use a subset for simplicity — modify these as needed
features = ['payment_value', 'price', 'freight_value']
target = 'payment_value'

# Remove NaN or invalid rows
data = data.dropna(subset=features)

# Scale features
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[features])

# Split data
X = []
y = []

# Create sequences (for LSTM)
sequence_length = 10
for i in range(sequence_length, len(data_scaled)):
    X.append(data_scaled[i-sequence_length:i])
    y.append(data_scaled[i, 0])  # predicting 'payment_value'

X, y = np.array(X), np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"✅ Training samples: {X_train.shape}, Testing samples: {X_test.shape}\n")

# ---------------------------------------------------------------------
# ✅ Build LSTM model
# ---------------------------------------------------------------------
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# ---------------------------------------------------------------------
# ✅ Train model
# ---------------------------------------------------------------------
print("🚀 Training the LSTM model...\n")
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1, verbose=1)

# ---------------------------------------------------------------------
# ✅ Evaluate
# ---------------------------------------------------------------------
loss = model.evaluate(X_test, y_test)
print(f"\n✅ Model evaluation complete. Test Loss: {loss:.6f}")

# ---------------------------------------------------------------------
# ✅ Save model and scaler
# ---------------------------------------------------------------------
model.save("lstm_model.h5")
import pickle
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n🎉 Model and scaler saved successfully!")
print("📦 Files created: lstm_model.h5, scaler.pkl")
