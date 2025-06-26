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
#plt.savefig('9_first_pictures.png')
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

''' ZAKOMENTOWANE POD PART 3
# 5. Kompilacja i trenowanie modelu

model.compile(optimizer="adam", metrics=["accuracy"], loss="sparse_categorical_crossentropy")
model.fit(X_train, y_train, epochs=3, validation_data= (X_val, y_val))

#zapis modelu do pliku
model.save("model.h5")


# 6. Obliczenie metryk i predykcje

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print(f"\nTest accuracy: {test_acc:.4f}")
print(f"\nTest loss: {test_loss:.4f}")

predictions = model.predict(X_test)
'''
# 7. Obliczenie metryk i predykcje II -- PATRZ PART-2
'''Stwórz kilka wykresów z predykcjami -
tj. takich, które będą wyświetlały obrazek ze zbioru testowego i przypisywaną mu klasę.
Wykres powinien być w formacie 3x3.'''

'''import numpy as np

# Zamiana predykcji z wektorów softmax na etykiety (klasy 0–9)
predicted_classes = np.argmax(predictions, axis=1)

# Wykresy 3x3 z predykcjami
plt.figure(figsize=(10, 10))

for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap="gray")
    pred_label = class_names[predicted_classes[i]]
    true_label = class_names[y_test[i]]
    title_color = "green" if predicted_classes[i] == y_test[i] else "red"
    plt.title(f"Pred: {pred_label}\nTrue: {true_label}", color=title_color)
    plt.axis("off")

plt.tight_layout()
plt.savefig("predictions_chart.png")
# plt.show()
'''

# 8. Tuning hiperparametrów

def build_model(hp):
    model = keras.Sequential()

    model.add(
        layers.Conv2D(
            filters=hp.Choice("conv_1_filter", values=[32, 64, 128]),
            kernel_size=hp.Choice("conv_1_kernel", values=[3, 5]),
            activation="leaky_relu",
            input_shape=(28, 28, 1)
        )
    )

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(
        layers.Conv2D(
            filters=hp.Choice("conv_2_filter", values=[32, 64]),
            kernel_size=hp.Choice("conv_2_kernel", values=[3, 5]),
            activation="leaky_relu"
        )
    )

    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())

    model.add(
        layers.Dense(
            units=hp.Int("dense_units", min_value=32, max_value=128, step=32),
            activation="leaky_relu"
        )
    )

    model.add(
        layers.Dropout(
            hp.Float("drop", min_value=0.2, max_value=0.5, step=0.1)
        )
    )

    model.add(layers.Dense(10, activation="softmax"))

    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model






'''patrz part-1'''


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