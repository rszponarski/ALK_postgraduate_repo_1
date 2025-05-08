"""macierz korelacji- tabela, która zawiera wspolczynniki korelacji dla różnych zmiennych w zestawie danych
-> chcemy wiedzieć jak zmienne są ze sobą powiązane"""

import pandas as pd
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

houses = pd.read_csv("houses.csv")
print(houses.isnull().sum())

num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

num_features = houses.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = houses.select_dtypes(include=['object']).columns.tolist()

houses[num_features] = num_imputer.fit_transform(houses[num_features])
houses[cat_features] = cat_imputer.fit_transform(houses[cat_features])
print(houses.isnull().sum())
plt.figure(figsize=(12,8))
sns.heatmap(houses[num_features].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidth=0.5)
plt.title("Korelacje cech w zbiorze Houses")
plt.savefig("heatmap_.png")

plt.figure(figsize=(8,6))
sns.histplot(houses["Cena"], bins=30, kde=True, color='blue')
plt.title('Histogram cechy')
plt.xlabel('Cena')
plt.ylabel("Ilość domów")
plt.savefig('histogram.png')

plt.figure(figsize=(10,6))
sns.boxplot(x=houses["Jakosc"], y=houses["Cena"], palette="coolwarm")
plt.title("Jakosc vs Cena")
plt.xlabel('Jakosc')
plt.ylabel("Cena")
plt.savefig('jakosc_vs_cena.png')

plt.figure(figsize=(8,6))
sns.scatterplot(x=houses['Powierzchnia'], y=houses['Cena'], color='purple', alpha=0.6)
plt.title("Powierzchnia vs Cena")
plt.xlabel('Powierzchnia')
plt.ylabel("Cena")
plt.savefig('powierzchnia_vs_cena.png')

plt.figure(figsize=(10,6))
sns.boxplot(x=houses["Cena"], palette="coolwarm")
plt.title("Cena")
plt.savefig('cena.png')

plt.figure(figsize=(10,6))
sns.boxplot(x=houses["Sasiedztwo"], y=houses["Cena"], palette="coolwarm")
plt.title("Sasiedztwo vs Cena")
plt.xlabel('Sąsiedztwo')
plt.xticks(rotation=90) # nazwy na osi x niewidoczne, więc obracamy
plt.ylabel("Cena")
plt.savefig('sasiedztwo_vs_cena_2.png')