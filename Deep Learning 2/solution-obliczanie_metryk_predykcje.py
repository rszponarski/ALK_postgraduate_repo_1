'''Usuń lub zakomentuj część kodu odpowiedzialną za tworzenie i trenowanie modelu.
Zamiast tego wczytaj gotowy model model.h5 do zmiennej model.

Dokonaj ewaluacji modelu na danych testowych.
Stwórz zmienne test_loss, test_acc i zapisz do nich odpowiednio wartość funkcji straty i accuracy.
PATRZ part1'''

import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# 1. Wstępne przygotowanie danych
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.8, random_state=42)

# 2. Wstępne przygotowanie danych cd.
X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_val = X_val.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# 3. Dodanie etykiet i eksploracja danych (opcjonalna wizualizacja)
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

'''plt.figure(figsize=(8,8))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(X_train[i], cmap="gray")
    plt.title(class_names[y_train[i]])
    plt.axis("off")
plt.tight_layout()
plt.savefig('9_first_pictures.png')
# plt.show()'''

# 4. Budowa i trening modelu (ZAKOMENTOWANE zgodnie z treścią zadania)

"""
model = models.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="leaky_relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="leaky_relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="leaky_relu"),
        layers.Dense(10, activation="softmax")
    ]
)

# 5. Kompilacja i trenowanie modelu

model.compile(optimizer="adam", metrics=["accuracy"], loss="sparse_categorical_crossentropy")
model.fit(X_train, y_train, epochs=3, validation_data=(X_val, y_val))
model.save("model.h5")
"""

# Wczytanie gotowego modelu (zgodnie z treścią zadania)
model = load_model("model.h5")

# 6. Obliczenie metryk i predykcje

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print(f"\nTest accuracy: {test_acc:.4f}")
print(f"\nTest loss: {test_loss:.4f}")

predictions = model.predict(X_test)

'''# (opcjonalne) drukowanie wyników — tylko do debugowania
print("Test accuracy:", test_acc)
print("Test loss:", test_loss)'''

