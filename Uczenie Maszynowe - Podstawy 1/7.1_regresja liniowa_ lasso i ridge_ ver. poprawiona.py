"""Użyj słowników models i metrics, które są już przygotowane.
Dodaj parametryzację modeli Ridge i Lasso, tak, aby alpha wynosiła 1.0.

Napisz pętlę po obiektach zawartych w models, która będzie:
- trenowała model (za pomocą odpowiedniej metody, którą odnajdziesz w dokumentacji dowolnego modelu znajdującego się w słowniku);
- zapisywała predykcje do słownika predictions pod odpowiednim kluczem (nazwą modelu ze słownika models);
- zapisywała nazwę modelu oraz obliczone metryki MAE (mean absolute error), RMSE (root mean square error) i R2 (R squared) do słownika metrics do listy będącej wartością odpowiedniego klucza;
- zapisywała współczynniki modelu do słownika coefficients pod odpowiednim kluczem (wartości współczynników wytrenowanego modelu możesz uzyskać odwołując się do odpowiedniego atrybutu danego modelu).

Pamiętaj, że model dopasowujemy na danych treningowych, natomiast prognozujemy wartości i przeprowadzamy walidację modelu na danych testowych.
Aby łatwo obliczyć wymienione wyżej metryki, skorzystaj z funkcji znajdujących się w module sklearn.metrics. """

import pandas as pd
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

houses = pd.read_csv("houses.csv")

num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

num_features = houses.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = houses.select_dtypes(include=['object']).columns.tolist()

houses[num_features] = num_imputer.fit_transform(houses[num_features])
houses[cat_features] = cat_imputer.fit_transform(houses[cat_features])

"""
Usuń ze zbioru houses obserwacje, których cena znajduje się poniżej co najmniej 1,5 IQR poniżej pierwszego kwartyla
i 1,5 IQR powyżej trzeciego kwartyla.
Wartości graniczne przypisz do zmiennych o nazwach lower_bound i upper_bound odpowiednio."""

Q1 = houses["Cena"].quantile(0.25)
Q3 = houses["Cena"].quantile(0.75)
IQR = Q3 - Q1  # rozstęp międzykwartylowy

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

houses = houses[(houses["Cena"] >= lower_bound) & (houses["Cena"] <= upper_bound)]

'''Encodowanie danych kategorycznych'''

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded = pd.DataFrame(encoder.fit_transform(houses[cat_features]))
encoded.index = houses.index

encoded.columns = encoder.get_feature_names_out(cat_features)

houses = houses.drop(columns=cat_features).join(encoded)

'''Podział danych na traningowe i testowe'''

X = houses.drop(columns=["Cena"])
y = houses["Cena"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

'''Skalowanie danych'''

scaler = StandardScaler()
num_features.remove('Cena')

X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])

# Skalowanie danych
scaler = StandardScaler()

X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])

# Lista modeli do przetestowania
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=1.0),
}

# Słowniki do przechowywania wyników
metrics = {"Model": [], "MAE": [], "RMSE": [], "R² Score": []}
predictions = {name: None for name in models}
coefficients = {name: None for name in models}

# Pętla po modelach
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    predictions[name] = y_pred

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    metrics["Model"].append(name)
    metrics["MAE"].append(mae)
    metrics["RMSE"].append(rmse)
    metrics["R² Score"].append(r2)

    coefficients[name] = model.coef_
