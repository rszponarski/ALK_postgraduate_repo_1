"""Logarytmowanie zamiast odcinania outlierów

Logarytmowanie jest często stosowane, by:
    -zmniejszyć wpływ wartości odstających,
    -uzyskać bardziej symetryczny rozkład (zbliżony do normalnego),
    -poprawić działanie modeli statystycznych czy uczenia maszynowego.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

# Wczytanie danych
houses = pd.read_csv("houses.csv")

# Imputacja braków
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

num_features = houses.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = houses.select_dtypes(include=['object']).columns.tolist()

houses[num_features] = num_imputer.fit_transform(houses[num_features])
houses[cat_features] = cat_imputer.fit_transform(houses[cat_features])

'''
# Heatmapa korelacji
plt.figure(figsize=(12,8))
sns.heatmap(houses[num_features].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidth=0.5)
plt.title("Korelacje cech w zbiorze Houses")
plt.savefig("heatmap_.png")

# Histogram oryginalnych danych
plt.figure(figsize=(8,6))
sns.histplot(houses["Cena"], bins=30, kde=True, color='blue')
plt.title('Histogram cechy - Cena (oryginalna)')
plt.xlabel('Cena')
plt.ylabel("Ilość domów")
plt.savefig('histogram.png')'''

# Logarytmowanie ceny
houses["Cena_log"] = np.log1p(houses["Cena"])  # log(1 + x)

print(houses.shape)

# Boxplot po logarytmowaniu
plt.figure(figsize=(10,6))
sns.boxplot(x=houses["Cena_log"], palette="coolwarm")
plt.title("Cena (po logarytmowaniu)")
plt.savefig('cena_log_boxplot.png')

# Histogram po logarytmowaniu
plt.figure(figsize=(8,6))
sns.histplot(houses["Cena_log"], bins=30, kde=True, color='blue')
plt.title('Histogram cechy - log(Cena)')
plt.xlabel('log(Cena)')
plt.ylabel("Ilość domów")
plt.savefig('histogram_log_cena.png')

print(houses[['Cena', 'Cena_log']].head())
# Odwrócenie logarytmowania
houses["Cena_log"] = np.expm1(houses["Cena_log"])

print(houses[['Cena', 'Cena_log']].head())
