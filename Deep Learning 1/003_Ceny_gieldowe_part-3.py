import keras
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import mean_squared_error
from keras_tuner import RandomSearch
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# 1. Przygotowanie danych
keras.utils.set_random_seed(43)
tf.config.experimental.enable_op_determinism()

df = pd.read_csv("sp500.txt")
df = df.sort_values("Date")
data = df[["Close"]].values

train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 60
X_train, y_train = create_sequences(train_scaled, seq_length)
X_test, y_test = create_sequences(test_scaled, seq_length)

# Upewnij się, że dane mają właściwy kształt
if X_train.ndim == 2:
    X_train = np.expand_dims(X_train, -1)
if X_test.ndim == 2:
    X_test = np.expand_dims(X_test, -1)

# --- Modele budowane dynamicznie przez Keras Tuner ---

def build_lstm_model(hp):
    model = Sequential()
    model.add(LSTM(hp.Int('lstm1', 32, 128, step=32), return_sequences=True, input_shape=(seq_length, 1)))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout1', 0.1, 0.5, step=0.1)))
    model.add(LSTM(hp.Int('lstm2', 32, 128, step=32)))
    model.add(Dropout(hp.Float('dropout2', 0.1, 0.5, step=0.1)))
    model.add(Dense(1))
    model.compile(optimizer=Adam(), loss='mse')
    return model

def build_gru_model(hp):
    model = Sequential()
    model.add(GRU(hp.Int('gru1', 32, 128, step=32), return_sequences=True, input_shape=(seq_length, 1)))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout1', 0.1, 0.5, step=0.1)))
    model.add(GRU(hp.Int('gru2', 32, 128, step=32)))
    model.add(Dropout(hp.Float('dropout2', 0.1, 0.5, step=0.1)))
    model.add(Dense(1))
    model.compile(optimizer=Adam(), loss='mse')
    return model

# Podział danych na treningowe i walidacyjne (10%)
val_split = int(len(X_train) * 0.1)
X_val, y_val = X_train[-val_split:], y_train[-val_split:]
X_train2, y_train2 = X_train[:-val_split], y_train[:-val_split]

# --- TUNING HIPERPARAMETRÓW (zakomentowany po zapisaniu modeli) ---
# tuner_lstm = RandomSearch(
#     build_lstm_model,
#     objective='val_loss',
#     max_trials=3,
#     executions_per_trial=1,
#     directory="lstm_tuning",
#     project_name="lstm",
#     seed=42
# )
# tuner_lstm.search(X_train2, y_train2, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=1)
# best_lstm = tuner_lstm.get_best_models(1)[0]
# best_lstm.save("best_lstm.h5")

# tuner_gru = RandomSearch(
#     build_gru_model,
#     objective='val_loss',
#     max_trials=3,
#     executions_per_trial=1,
#     directory="gru_tuning",
#     project_name="gru",
#     seed=42
# )
# tuner_gru.search(X_train2, y_train2, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=1)
# best_gru = tuner_gru.get_best_models(1)[0]
# best_gru.save("best_gru.h5")

# --- WCZYTANIE WCZEŚNIEJ ZAPISANYCH MODELI ---
best_lstm = load_model("best_lstm.h5")
best_gru = load_model("best_gru.h5")

# --- Przewidywanie i ewaluacja ---
pred_lstm = scaler.inverse_transform(best_lstm.predict(X_test))
pred_gru = scaler.inverse_transform(best_gru.predict(X_test))
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

rmse_lstm = np.sqrt(mean_squared_error(y_test_inv, pred_lstm))
rmse_gru = np.sqrt(mean_squared_error(y_test_inv, pred_gru))

print("RMSE LSTM:", rmse_lstm)
print("RMSE GRU:", rmse_gru)

# --- Wykres porównania ---
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label="Rzeczywiste")
plt.plot(pred_lstm, label="LSTM")
plt.plot(pred_gru, label="GRU")
plt.legend()
plt.title("Porównanie modeli: LSTM vs GRU")
plt.savefig("porownanie_modeli_ROC-2.png")
