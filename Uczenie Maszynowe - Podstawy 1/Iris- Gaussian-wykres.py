import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process.kernels import RBF

# 1. Wczytanie danych
df = pd.read_csv("iris.csv")

# 2. Wybór dwóch cech (petal_length i petal_width)
X = df[["petal_length", "petal_width"]].values
y = df["species"].values  # nazwy gatunków: setosa, versicolor, virginica

# 3. Enkodowanie etykiet
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)  # np. 0, 1, 2

# 4. Skalowanie danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=0)

# 6. Trening modelu GaussianProcessClassifier
kernel = 1.0 * RBF(length_scale=1.0)
model_gaussian = GaussianProcessClassifier(kernel=kernel, random_state=0)
model_gaussian.fit(X_train, y_train)

# 7. Tworzenie siatki do predykcji (meshgrid)
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

# 8. Predykcja prawdopodobieństw dla każdego punktu siatki
Z = model_gaussian.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape((300, 300, -1))

# 9. Wykres
plt.figure(figsize=(10, 6))

# Kolorowe tło z prawdopodobieństwami dla każdej klasy
for i in range(3):  # 3 klasy
    plt.contourf(xx, yy, Z[:, :, i], alpha=0.3, levels=10, cmap='viridis')

# Granice decyzyjne
Z_labels = model_gaussian.predict(np.c_[xx.ravel(), yy.ravel()])
Z_labels = Z_labels.reshape(xx.shape)
plt.contour(xx, yy, Z_labels, levels=[0.5, 1.5], linewidths=2, colors='black')

# Punkty treningowe
for class_index, class_name in enumerate(encoder.classes_):
    plt.scatter(X_scaled[y_encoded == class_index, 0],
                X_scaled[y_encoded == class_index, 1],
                label=class_name)

plt.title("GaussianProcessClassifier – Petal Length vs Width")
plt.xlabel("petal_length (scaled)")
plt.ylabel("petal_width (scaled)")
plt.legend()
plt.colorbar(label="Prawdopodobieństwo")
plt.grid(True)
plt.savefig('Gaussian Classifier_petal length and width.png')

