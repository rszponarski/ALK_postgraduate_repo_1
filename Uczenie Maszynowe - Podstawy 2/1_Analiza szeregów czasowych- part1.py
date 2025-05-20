import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Wczytanie i wstępne przygotowanie danych
# --- Ważne jest, by ramka danych -bitcoin- była indeksowana datą.

bitcoin = pd.read_csv('bitcoin.csv', parse_dates=['date'], dayfirst=True, index_col='date')
bitcoin['value'] = pd.to_numeric(bitcoin['value'].str.replace(",", "."))
print(bitcoin['value'])

# Wykres szeregu czasowego
# -- przygotowanie wykresu przedstawiający bitcoin["value"] w czasie

plt.figure(figsize=(12,6))
plt.plot(bitcoin.index, bitcoin['value'], label="Cena Bitcoina", color='purple')
plt.xlabel('Data')
plt.ylabel("Cena")
plt.title('Cena Bitcoina w czasie')
plt.legend()
plt.grid(True)  # Dodaje siatkę
plt.savefig('solution.png')

# Feature engineering: dane czasowe I
# -- Stwórz w ramce danych następujące kolumny: year, month, day, weekday
# -- w kolumnie weekday powinny znajdować się numery dni tygodnia, nie ich nazwy.

bitcoin['year'] = bitcoin.index.year
bitcoin['month'] = bitcoin.index.month
bitcoin['day'] = bitcoin.index.day
bitcoin['weekday'] = bitcoin.index.weekday

# Feature engineering: dane czasowe II
# reprezentacja miesiąca i dnia tygodnia jako wartości czysto liczbowych (styczeń (1) i grudzień (12) są obok siebie),
# stworzenie w ramce danych bitcoin następujące kolumny: month_sin, month_cos, weekday_sin, weekday_cos.

bitcoin['month_sin'] = np.sin(2 * np.pi * bitcoin["month"] / 12)
bitcoin['month_cos'] = np.cos(2 * np.pi * bitcoin["month"] / 12)
bitcoin['weekday_sin'] = np.sin(2 * np.pi * bitcoin["weekday"] / 7)
bitcoin['weekday_cos'] = np.cos(2 * np.pi * bitcoin["weekday"] / 7)

# Feature engineering: dane czasowe III

""" W ramce danych nie mamy jak do tej pory zmiennej objaśnianej i objaśniających, tylko jedną zmienną.
W modelowaniu szeregów czasowych najpowszechniejsze są dwa typy modeli:
  --- AR (oparty na opóźnieniach zmiennej)
  --- ARMA (oparty na opóźnieniach zmiennej i średniej ruchomej).

Stworzymy dla opóźnień: 1, 3, 7, 14 kolumny bitcoin[f'value_lag_{lag}'] i bitcoin[f'value_roll_mean_{lag}'],
gdzie lag to stopień opóźnienia. """

for lag in [1, 3, 7, 14]:
    bitcoin[f"value_lag_{lag}"] = bitcoin["value"].shift(lag)
    bitcoin[f"value_roll_mean_{lag}"] = bitcoin["value"].rolling(window=lag).mean().shift(lag)

bitcoin.dropna(inplace=True)

# Wstęp do cross-validation
# stworzenie zmiennych X i y; inicjacja obiektu o nazwie ts_split typu TimeSeriesSplit z n_splits=5.

X = bitcoin.drop(columns='value')
y = bitcoin['value']

ts_split = TimeSeriesSplit(n_splits=5)

models = {
    'LinearRegression': LinearRegression(),
    'Lasso (alpha: 0.1)': Lasso(alpha=0.1, random_state=42),
    'SVR': SVR(C=3, kernel='rbf', epsilon=0.1),
    'KNeighbors': KNeighborsRegressor(n_neighbors=5, weights='distance'),
    'XGBoost': XGBRegressor(max_depth=4, subsample=0.8, random_state=42),
    'RandomForest': RandomForestRegressor(max_depth=4, random_state=42)
}

# Modelowanie z cross-validation
"""Modelowanie w pętli; pętla podwójna:
 -- zewnętrzna po modelach,
 -- wewnętrzna po kolejnych próbkach zestawów danych. """

#  --- Inicjalizacja słowników do wyników i predykcji

results = {name: {"R²": [], "MAE": [], "RMSE": []} for name in models}
predictions = {name: [] for name in models}

# Skaler
scaler = StandardScaler()

# Trenowanie modeli i ewaluacja
for name, model in models.items():
    for i, (train_index, test_index) in enumerate(ts_split.split(X)):
        # Podział na dane treningowe i testowe
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        if i==0:
            X_train_scaled = scaler.fit_transform(X_train)
        else:
            X_train_scaled = scaler.transform(X_train)

        X_test_scaled = scaler.transform(X_test)

        # ---------------------------------------
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        results[name]["R²"].append(r2_score(y_test, y_pred))
        results[name]["MAE"].append(mean_absolute_error(y_test, y_pred))
        results[name]["RMSE"].append(np.sqrt(mean_squared_error(y_test, y_pred)))
        predictions[name].append((bitcoin.index[test_index], y_pred))

# Modelowanie z cross-validation: porównanie wyników
# (Wciąż w pętli) dopasuj model do danych treningowych. Stwórz obiekt y_pred i przypisz do niego predykcje zbioru testowego.
# PATRZ pętla powyżej od (------)

# Modelowanie z cross-validation: porównanie wyników II
"""Ze względu na to, że każda klasa modelu (np. LinearRegression) tworzyła predykcje dla pięciu różnych próbek,
chcąc porównać te klasy, trzeba obliczyć średnią z metryk na tych próbkach."""

final_results = {name: {metric: np.mean(values) for metric, values in metrics.items()} for name, metrics in results.items()}
results_df = pd.DataFrame(final_results).T
# print(results_df)

# Wizualizacja wyników

def plot_predictions(model_name, predictions, y):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y.index, y, label='Wartości rzeczywiste', linestyle='dashed', color='black')

    for idx, y_pred in predictions:
        ax.plot(idx, y_pred, label=f'{model_name}: predykcje')

    ax.set_title(f'Wykres dla modelu {model_name}')
    ax.set_xlabel(f'Data')
    ax.set_ylabel('Cena Bitcoina')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{model_name}_wizualizacja_predykcje.png")

for name, preds in predictions.items():
    plot_predictions(name, preds, y)

# Poprawa metryk modelu
'''Teraz, gdy już na podstawie wykresów wiemy, z czego wynikają złe wyniki niektórych modeli,
możemy to zmienić i pominąć zapisywanie predykcji dla pierwszego próbkowanego okresu.'''


