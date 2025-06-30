import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from sklearn.metrics import accuracy_score

# 1. ===================================================================================
''' Wczytanie i początkowe przygotowanie danych '''

# Wczytanie danych
df = pd.read_csv("heart.csv")

# One-hot encoding dla kolumn kategorycznych
df = pd.get_dummies(df, columns=["Sex", "ST_Slope", "ExerciseAngina", "RestingECG", "ChestPainType"])

# 2. ===================================================================================
''' Podział danych '''

# Podział na cechy i etykietę
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. ===================================================================================
''' Skalowanie danych '''

# Skalowanie danych
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. ===================================================================================
''' Budowa modelu sekwencyjnego '''

# Budowa modelu – zakomentowana, ponieważ model już wytrenowany
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# model = Sequential()
# model.add(Dense(6, activation='relu', input_shape=(X_train.shape[1],)))
# model.add(Dense(6, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. ===================================================================================
''' Trening modelu '''

# history = model.fit(X_train, y_train, validation_split=0.33, batch_size=10, epochs=100)
# model.save("model.h5")

# 6. ===================================================================================
''' Predykcja na zbiorze testowym '''

# Wczytanie wytrenowanego modelu
model = tf.keras.models.load_model("model_dodatkowa_praktyka.h5")

# Predykcja na zbiorze testowym
predictions = model.predict(X_test)

# Konwersja wyników na wartości True/False
y_pred = [[float(p[0]) > 0.5] for p in predictions]

# 7. ===================================================================================
''' Ocena trafności modelu '''

# Obliczenie accuracy
accuracy = accuracy_score(y_test, y_pred)

# Wyświetlenie wyniku (opcjonalne)
print("Accuracy:", accuracy)