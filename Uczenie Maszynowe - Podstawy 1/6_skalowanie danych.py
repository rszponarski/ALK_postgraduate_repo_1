"""Użyj StandardScaler, aby dokonać standaryzacji danych na zbiorze treningowym i testowym.
Dane przeskalowane powinny nosić nazwy: X_train i X_test. """

import pandas as pd
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

houses = pd.read_csv("houses.csv")

num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

num_features = houses.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = houses.select_dtypes(include=['object']).columns.tolist()

houses[num_features] = num_imputer.fit_transform(houses[num_features])
houses[cat_features] = cat_imputer.fit_transform(houses[cat_features])

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

"""
Usuń ze zbioru houses obserwacje, których cena znajduje się poniżej co najmniej 1,5 IQR poniżej pierwszego kwartyla
i 1,5 IQR powyżej trzeciego kwartyla.
Wartości graniczne przypisz do zmiennych o nazwach lower_bound i upper_bound odpowiednio."""

Q1 = houses["Cena"].quantile(0.25)
Q3 = houses["Cena"].quantile(0.75)
IQR = Q3 - Q1  # rozstęp międzykwartylowy

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# print(houses[(houses["Cena"] <= lower_bound) | (houses["Cena"] >= upper_bound)])
print(houses[(houses["Cena"] <= lower_bound) | (houses["Cena"] >= upper_bound)].shape[0])

houses = houses[(houses["Cena"] >= lower_bound) & (houses["Cena"] <= upper_bound)]

plt.figure(figsize=(10,6))
sns.boxplot(x=houses["Cena"], palette="viridis", whis=1.5)
plt.title("Cena")
plt.savefig('cena_deleted_outliers.png')

plt.figure(figsize=(8,6))
sns.histplot(houses["Cena"], bins=30, kde=True, color='blue')
plt.title('Histogram cechy')
plt.xlabel('Cena')
plt.ylabel("Ilość domów")
plt.savefig('histogram_bez_outlier.png')

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