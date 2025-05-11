"""Do słownika models dodaj klucz "XGBoost" zawierający model XGBRegressor o funkcji celu "reg:squarederror".
Powinien mieć:
    100 estymatorów,
    maksymalną głębokość drzewa 6,
    współczynnik uczenia 0.1,
    wykorzystywać 80% próbki do zbudowania jednego drzewa;
    i mieć random_state=42.
Pamiętaj, że XGBoost nie ma współczynników, więc odniesienie do coefficients w pętli trzeba obudować warunkiem."""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import xgboost as xgb

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

# Modele
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=1.0),
    "XGBoost": xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
}

metrics = {"Model": [], "MAE": [], "RMSE": [], "R² Score": []}
predictions = {name: None for name in models}
coefficients = {name: None for name in models}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred

    metrics["Model"].append(name)
    metrics["MAE"].append(mean_absolute_error(y_test, y_pred))
    metrics["RMSE"].append(np.sqrt(mean_squared_error(y_test, y_pred)))
    metrics["R² Score"].append(r2_score(y_test, y_pred))

    if hasattr(model, "coef_"):
        coefficients[name] = model.coef_

#print(metrics)

'''Sprawdź, co kryje się w wypełnionym obiekcie metrics. Konwertuj go na data frame, nadając mu nazwę results_df.
Stwórz także zmienną importances, której przypiszesz współczynniki "ważności" cech modelu XGBoost. '''

results_df = pd.DataFrame(metrics)
#print(results_df)

# współczynniki "ważności" cech modelu XGBoost
importances = models["XGBoost"].feature_importances_
print(importances)

"""Wyświetlenie w przystępniejszej formie nazw cech i ich ważności"""

# Stworzenie DataFrame z nazwami kolumn i ich ważnością
feature_importance_df = pd.DataFrame({
    "Cecha": X_train.columns,
    "Ważność": importances
})

# Posortowanie od najważniejszej cechy
feature_importance_df = feature_importance_df.sort_values(by="Ważność", ascending=False)
#print(feature_importance_df)

"""Sporządź wykres typu bar plot ważności cech z modelu XGBoost."""

plt.figure(figsize=(30,10))
plt.bar(range(len(importances)), importances, color='green')
plt.xticks(range(len(importances)), [X_train.columns[i] for i in range(len(importances))])
plt.xlabel("Cechy")
plt.ylabel("Importances")
plt.xticks(rotation=70)
plt.title("Feature importances: XGBoost")
plt.tight_layout()
plt.savefig('feature_importances_chart.png')

"""Sporządź cztery wykresy typu scatter plot, gdzie na osi X znajdą się wartości rzeczywiste, a na Y przewidywane
- po jednym dla każdego z przygotowanych modeli. """

def plot_model_vs_prediction(ax, model_name, y_true, y_pred):
    ax.scatter(y_true, y_pred, color='green', alpha=0.5)
    ax.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')
    ax.set_title(f"{model_name} - rzeczywiste vs przewidywane")
    ax.set_xlabel("Wartości rzeczywiste")
    ax.set_ylabel("Wartości przewidywane")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

plot_model_vs_prediction(axes[0, 0], "Linear Regression", y_test, predictions["Linear Regression"])
plot_model_vs_prediction(axes[0, 1], "Ridge Regression", y_test, predictions["Ridge Regression"])
plot_model_vs_prediction(axes[1, 0], "Lasso Regression", y_test, predictions["Lasso Regression"])
plot_model_vs_prediction(axes[1, 1], "XGBoost", y_test, predictions["XGBoost"])
plt.savefig("model_vs_prediction_comparison.png")