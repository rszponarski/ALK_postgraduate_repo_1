import keras
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# 1. Wstępne przygotowanie danych
keras.utils.set_random_seed(43)
tf.config.experimental.enable_op_determinism()

df = pd.read_csv("sp500.txt")
df = df.sort_values("Date")
data = df[["Close"]].values

train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

# 2. Skalowanie danych
'''Inicjalizacja MinMaxScaler z pakietu sklearn; zapisanie go pod nazwą scaler.
Skalowanie odpowiednio danych testowych i treningowych;
przypisanie do odpowiadających zmiennych: train_scaled i test_scaled.'''

scaler = MinMaxScaler()

train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

# 3. Tworzenie sekwencji danych czasowych

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# 4. Tworzenie sekwencji danych czasowych c.d.
'''Użyj funkcji create_sequences, aby stworzyć sekwencje X_train, y_train oraz X_test, y_test o długości 60. '''
seq_length = 60

X_train, y_train = create_sequences(train_scaled, seq_length)
X_test, y_test = create_sequences(test_scaled, seq_length)

# 5. Model LSTM

model_lstm = Sequential([
    LSTM(64, return_sequences=True, input_shape=(seq_length, 1)),
    BatchNormalization(),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])

model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
# model_lstm.fit(X_train, y_train, epochs=10, batch_size=32)  # Trenowanie modelu

# model_lstm.save("model_lstm.h5")  # Zapis modelu

