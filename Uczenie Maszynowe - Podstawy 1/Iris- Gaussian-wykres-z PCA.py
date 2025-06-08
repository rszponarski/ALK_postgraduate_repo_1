import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

# 1. Wczytanie danych
df = pd.read_csv("iris.csv")

X = df.drop("species", axis=1)
y = df["species"]

# 2. Enkodowanie etykiet
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# 3. Skalowanie danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. PCA - redukcja wymiarów do 2
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 5. Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_encoded, test_size=0.2, random_state=0)

# 6. Trening GaussianProcessClassifier
model_gaussian = GaussianProcessClassifier(random_state=0)
model_gaussian.fit(X_train, y_train)

# 7. Predykcja
y_pred = model_gaussian.predict(X_test)
y_pred_labels = encoder.inverse_transform(y_pred)

# 8. Wizualizacja
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]
probs = model_gaussian.predict_proba(grid)
Z = np.max(probs, axis=1).reshape(xx.shape)
Z_labels = model_gaussian.predict(grid).reshape(xx.shape)

# 9. Rysowanie wykresu
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap='viridis', alpha=0.4)
plt.contour(xx, yy, Z_labels, levels=[0.5, 1.5], colors='k')

# Punkty danych
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_encoded, cmap='tab10', edgecolor='k', s=40)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('GaussianProcessClassifier – PCA 2D')
plt.legend(handles=scatter.legend_elements()[0], labels=list(encoder.classes_), title="Klasy")
plt.colorbar(label="Prawdopodobieństwo")
plt.grid(True)
plt.savefig('Gaussian Classifier_with PCA.png')

