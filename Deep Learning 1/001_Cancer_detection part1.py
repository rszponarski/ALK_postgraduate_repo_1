import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Wyświetl wszystkie kolumny
# pd.set_option('display.max_columns', None)

# 1. Wstępne przygotowanie danych
dataset = pd.read_csv("cancer.csv")

X = dataset.drop(["diagnosis", "id"], axis=1)
y = dataset["diagnosis"]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 2. Wstępne przygotowanie danych cd.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Budowanie modelu

model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation="relu", input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# 4. Kompilacja modelu
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 5. Trening modelu
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=30,
    validation_split=0.1,
    verbose=1
)

# 6. Predykcje

preds = model.predict(X_test).reshape(-1)
preds_binary = (preds > 0.5).astype(int)