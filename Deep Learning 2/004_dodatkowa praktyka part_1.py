import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. ===================================================================================
# Wczytanie danych
df = pd.read_csv("heart.csv")

print(df.head()) # Wyświetlenie pierwszych kilku wierszy, by sprawdzić dane

# Przekształcenie kolumn kategorycznych na zmienne zero-jedynkowe (one-hot encoding)
df = pd.get_dummies(df, columns=["Sex", "ST_Slope", "ExerciseAngina", "RestingECG", "ChestPainType"])

print(df.head()) # Wyświetlenie wynikowego DataFrame


# 2. ===================================================================================
# Podział danych na X (cechy) i y (etykieta)
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Podział danych na zbiory treningowy i testowy (80:20), z zachowaniem losowości
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Opcjonalnie: sprawdzenie kształtów zbiorów
#print("X_train:", X_train.shape)
#print("X_test:", X_test.shape)
#print("y_train:", y_train.shape)
#print("y_test:", y_test.shape)


# 3. ===================================================================================
# Tworzenie obiektu skalera
scaler = StandardScaler()

# Dopasowanie skalera do danych treningowych i transformacja
X_train = scaler.fit_transform(X_train)

# Transformacja danych testowych (bez ponownego dopasowania!)
X_test = scaler.transform(X_test)

# 4. ===================================================================================
# Tworzenie modelu sekwencyjnego
model = Sequential()

# Warstwa wejściowa
model.add(Dense(6, activation='relu', input_shape=(X_train.shape[1],)))

# Warstwa ukryta
model.add(Dense(6, activation='relu'))

# Warstwa wyjściowa
model.add(Dense(1, activation='sigmoid'))

# Kompilacja modelu
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Wyświetlenie podsumowania modelu
model.summary()

# 5. ===================================================================================
# Trenowanie modelu
history = model.fit(
    X_train,
    y_train,
    validation_split=0.33,  # część zbioru treningowego przeznaczona na walidację
    batch_size=10,          # wielkość próbki
    epochs=100,             # liczba epok
    verbose=1               # pokazuj postęp treningu
)

# Zapisanie modelu do pliku model.h5
model.save("model_dodatkowa_praktyka.h5")


