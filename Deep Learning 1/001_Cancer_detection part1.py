import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

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

# 7. Ocena modelu

cr = classification_report(y_test, preds_binary, target_names=label_encoder.classes_)
cm = confusion_matrix(y_test, preds_binary)

# 8. Macierz pomyłek
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig("confusion_matrix_cancer_detection.png")

# 9. Wizualizacja przebiegu treningu

plt.figure(figsize=(8, 6))
plt.plot(history.history["accuracy"], label = "Train accuracy")
plt.plot(history.history["val_accuracy"], label = "Val accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Model training accuracy")
plt.savefig("model_training_accuracy.png")
