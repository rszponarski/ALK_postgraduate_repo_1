import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

'''
keras.utils.set_random_seed(43)
tf.config.experimental.enable_op_determinism()'''

# Wyświetl wszystkie kolumny
# pd.set_option('display.max_columns', None)


# 1. Wstępne przygotowanie danych
fashion_mnist = tf.keras.datasets.fashion_mnist

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.8, random_state=42)


# 2. Wstępne przygotowanie danych cd.

''''(liczba_obrazów, wysokość, szerokość, liczba_kanałów)'''
X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_val = X_val.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)


# 3. Dodanie etykiet i eksploracja danych

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
    ]

plt.figure(figsize=(8,8))

for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(X_train[i], cmap="gray")
    plt.title(class_names [y_train[i]])
    plt.axis("off")

plt.tight_layout()
plt.savefig('9_first_pictures.png')
#plt.show()


# 4. Budowa modelu z CNN

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
model.fit(X_train, y_train, epochs=3, validation_data= (X_val, y_val))

'''zapis modelu do pliku'''
model.save("model.h5")


# 6. Obliczenie metryk i predykcje

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print(f"\nTest accuracy: {test_acc:.4f}")
print(f"\nTest loss: {test_loss:.4f}")

predictions = model.predict(X_test)

'''---- brudnopis ----
??? leaky_relu nie działa jako string. ??? Trzeba go używać jako osobnej warstwy LeakyReLU().

W Kerasie możesz podać nazwę funkcji aktywacji jako string (np. "relu", "sigmoid", "softmax", itd.),
ale "leaky_relu" nie jest rozpoznawany jako string.
To dlatego, że LeakyReLU to osobna warstwa, nie tylko funkcja.

Przykład poprawnej definicji modelu:

model = models.Sequential([
    layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1)),
    layers.LeakyReLU(),                         # <- osobna warstwa aktywacji
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3)),
    layers.LeakyReLU(),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64),
    layers.LeakyReLU(),

    layers.Dense(10, activation="softmax")      # <- softmax może być jako string
])'''