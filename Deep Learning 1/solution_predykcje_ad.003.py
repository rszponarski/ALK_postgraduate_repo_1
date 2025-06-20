import keras
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import mean_squared_error

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

# Upewnij się, że X_test ma odpowiedni kształt (samples, timesteps, features)
if X_test.ndim == 2:
    X_test = np.expand_dims(X_test, -1)

# 2. Wczytanie modeli
model_lstm = tf.keras.models.load_model("model_lstm.h5")
model_gru = tf.keras.models.load_model("model_gru.h5")

# 3. Przewidywania
pred_lstm = scaler.inverse_transform(model_lstm.predict(X_test))
pred_gru = scaler.inverse_transform(model_gru.predict(X_test))
y_test_inv = scaler.inverse_transform(y_test)

# 4. RMSE
rmse_lstm = np.sqrt(mean_squared_error(y_test_inv, pred_lstm))
rmse_gru = np.sqrt(mean_squared_error(y_test_inv, pred_gru))

print("RMSE LSTM: ", rmse_lstm)
print("RMSE GRU: ", rmse_gru)
