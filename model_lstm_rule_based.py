import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense

def train_lstm_model(data):
    X = data[["pressure", "flow", "speed"]].values.reshape(-1, 1, 3)
    y = data["failure"].values

    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(1, 3)),
        LSTM(30, activation='relu'),
        Dense(1, activation='sigmoid')  
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=10, batch_size=32, verbose=1)
    
    return model

def predict_lstm(data, model):
    X = data[["pressure", "flow", "speed"]].values.reshape(-1, 1, 3)
    predictions = (model.predict(X) > 0.6).astype(int)  # Podniesiony próg predykcji do 0.6
    data["predicted_failure"] = predictions
    return data

data = pd.read_csv("Data/actuator_data.csv")
model = train_lstm_model(data)
lstm_results = predict_lstm(data, model)
lstm_results.to_csv("Data/lstm_results.csv", index=False)
print("LSTM model results saved.")
