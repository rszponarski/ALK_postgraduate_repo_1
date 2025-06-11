import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Wyświetl wszystkie kolumny
# pd.set_option('display.max_columns', None)

# 1. Wstępne przygotowanie danych
df = pd.read_csv("creditcard.csv")

X = df.drop(["Time", "Class"], axis=1).values
y = df["Class"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Podział danych na treningowe i testowe
X_train = X_scaled[y == 0]
X_test= X_scaled

# 3. Budowa enkodera

input_dim = X_train.shape[1]
encoding_dim = 14

input_layer = Input(shape=(input_dim))
encoded = Dense(32, activation="relu")(input_layer)
encoded = BatchNormalization()(encoded)
encoded = Dropout(0.2)(encoded)

encoded = Dense(encoding_dim, activation="relu")(encoded)