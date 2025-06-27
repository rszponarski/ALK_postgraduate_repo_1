import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.datasets import fashion_mnist

# 1. Załaduj dane testowe
(_, _), (X_test, y_test) = fashion_mnist.load_data()

X_test = X_test / 255.0
X_test = X_test.reshape(-1, 28, 28, 1)

# 2. Załaduj model z odpowiednią aktywacją
model = load_model("best_model.h5", custom_objects={"leaky_relu": LeakyReLU})

# 3. Ewaluacja
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_acc}")

# 4. Predykcje
predictions = model.predict(X_test)
