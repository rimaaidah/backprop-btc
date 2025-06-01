from flask import Flask, render_template, request
from keras.models import load_model
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

model = load_model('model_btc.h5')
scaler = joblib.load('scaler_btc.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    autofill = {}

    # Baca BTC.csv
    csv_path = 'BTC.csv'
    df_table = None
    if os.path.exists(csv_path):
        df_table = pd.read_csv(csv_path)
        df_table = df_table.head(10)  # Hanya tampilkan 10 baris pertama biar ringkas

    if request.method == 'POST':
        open_val = float(request.form['open'])
        high_val = float(request.form['high'])
        low_val = float(request.form['low'])
        volume_val = float(request.form['volume'])

        data = np.array([[open_val, high_val, low_val, volume_val]])
        scaled_data = scaler.transform(data)
        pred_scaled = model.predict(scaled_data)
        prediction = pred_scaled[0][0]

    return render_template("index.html", prediction=prediction, autofill=autofill, table=df_table)

if __name__ == '__main__':
    app.run(debug=True)
