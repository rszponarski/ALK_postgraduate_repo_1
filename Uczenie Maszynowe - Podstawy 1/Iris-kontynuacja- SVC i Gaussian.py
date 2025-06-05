import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier

# 1. Wczytanie danych
df = pd.read_csv("iris.csv")
X = df.drop("species", axis=1)
y = df["species"]

# 2. Kodowanie etykiet
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)  # np. setosa -> 0, versicolor -> 1, virginica -> 2

# 3. Skalowanie danych
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)

# 4. Podział na dane treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y_encoded, test_size=0.2, random_state=0
)

# 5. Trening i predykcja modelu KNN
model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn.fit(X_train, y_train)
y_pred_knn_encoded = model_knn.predict(X_test)
y_pred_knn = encoder.inverse_transform(y_pred_knn_encoded)

# 6. Trening i predykcja modelu SVC
model_svc = SVC(random_state=0)
model_svc.fit(X_train, y_train)
svc_pred_encoded = model_svc.predict(X_test)
svc_pred = encoder.inverse_transform(svc_pred_encoded)

# 7. Trening i predykcja modelu GaussianProcessClassifier
model_gaussian = GaussianProcessClassifier(random_state=0)
model_gaussian.fit(X_train, y_train)
gaussian_pred_encoded = model_gaussian.predict(X_test)
gaussian_pred = encoder.inverse_transform(gaussian_pred_encoded)

# 8. Przykładowe wyniki
print("Przykładowe predykcje KNN:", y_pred_knn[:5])
print("Przykładowe predykcje SVC:", svc_pred[:5])
print("Przykładowe predykcje Gaussian:", gaussian_pred[:5])

