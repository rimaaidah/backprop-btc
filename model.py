# model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# 1. Load dan normalisasi nama kolom
df = pd.read_csv("BTC.csv")
df.columns = df.columns.str.lower()  # ubah semua kolom jadi lowercase

# 2. Pastikan kolom yang dibutuhkan ada
required_cols = ['open', 'high', 'low', 'close', 'volume']
missing = [col for col in required_cols if col not in df.columns]
if missing:
    raise ValueError(f"Kolom berikut tidak ditemukan di CSV: {missing}")

# 3. Pilih kolom dan proses
df = df[required_cols].dropna()
X = df[['open', 'high', 'low', 'volume']].values
y = df[['close']].values

# 4. Normalisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 6. Model ANN
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mean_squared_error')

# 7. Training dengan EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_train, y_train, epochs=200, batch_size=16, validation_data=(X_test, y_test), callbacks=[early_stop])

# 8. Simpan model dan scaler
model.save("model_btc.h5")
joblib.dump(scaler, "scaler_btc.pkl")

print("Model dan scaler berhasil disimpan sebagai 'model_btc.h5' dan 'scaler_btc.pkl'")
