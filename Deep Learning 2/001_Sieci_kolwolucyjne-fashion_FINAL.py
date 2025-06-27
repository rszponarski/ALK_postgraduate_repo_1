import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import keras_tuner as kt # Not needed for submission if tuning is commented out

keras.utils.set_random_seed(43)
tf.config.experimental.enable_op_determinism()

# Data loading and preprocessing (keep this)
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.8, random_state=42)

X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_val = X_val.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

# Plotting code (can be kept or commented out for submission, depending on requirements)
plt.figure(figsize=(8,8))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(X_train[i], cmap="gray")
    plt.title(class_names [y_train[i]])
    plt.axis("off")
plt.tight_layout()
#plt.show()

# The original model definition (not used if loading best_model)
# model = models.Sequential(...)

# Hyperparameter tuning function (keep the definition, but don't call the tuner)
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


"""# 5. Konfiguracja tunera (COMMENT OUT THIS SECTION)
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=3,
    executions_per_trial=1,
    overwrite=True,
    directory='tuner_dir',
    project_name='fashion_cnn'
)

# === UCZENIE: ODKOMENTUJ tylko na etapie trenowania === (COMMENT OUT THIS SECTION)
tuner.search(X_train, y_train, epochs=3, validation_data=(X_val, y_val))

best_model = tuner.get_best_models(num_models=1)[0]
best_model.fit(X_train, y_train, epochs=3, validation_data=(X_val, y_val))  # lub więcej epok
best_model.save("best_model.h5")
val_loss, val_acc = best_model.evaluate(X_val, y_val)
print("Validation accuracy:", val_acc)"""


# === ZALICZENIE: ODKOMENTUJ tylko na etapie testowania === (THIS IS THE SECTION TO UNCOMMENT)

best_model = keras.models.load_model("best_model.h5")
test_loss, test_acc = best_model.evaluate(X_test, y_test)
predictions = best_model.predict(X_test)

print("Test loss:", test_loss)
print("Test accuracy:", test_acc)


# 10. Macierz pomyłek

import seaborn as sns
from sklearn.metrics import confusion_matrix

y_pred = best_model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)

cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title('Macierz pomyłek')
plt.xlabel('Klasa przewidywania')
plt.ylabel('Klasa rzeczywista')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('confusion_matrix_module1.png')
