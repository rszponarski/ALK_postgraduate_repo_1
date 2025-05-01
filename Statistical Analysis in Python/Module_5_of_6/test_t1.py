# TASK 1 MODULE 5

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
# implementation of the k-means algorithm for grouping data into clusters.
import matplotlib.pyplot as plt

df = pd.read_csv("seeds_dataset.csv")
columns = df.columns

# Przypisanie kolumn pod późniejszy wykres
x, y = df[columns[5]], df[columns[6]]

# Standaryzacja danych
scaler = StandardScaler()
# StandardScaler przekształca dane tak, aby każda cecha miała średnią 0 i odchylenie standardowe 1.
df_scaled = scaler.fit_transform(df)
# Standaryzacja poprawia działanie algorytmów klastrowania (jak k-średnie,) wrażliwych na różnice w skalach danych.

# Grouping by k-means
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(df_scaled)
'''  KMeans:  Parametr n_clusters=3 oznacza, że dane zostaną podzielone na 3 klastry.
              random_state=10 ustala ziarno losowości, aby uzyskać powtarzalne wyniki.
fit_predict:  Dostosowuje model do danych (fit) i przypisuje etykiety klastrów do próbek (predict).
              clusters zawiera etykiety klastra przypisane do każdej próbki w danych. '''

# Wizualizacja: tak wygląda w oryginale:
plt.scatter(x, y, c=clusters, cmap="viridis")
# Wizualizacja: bez klastrów
# plt.scatter(x, y, cmap="viridis")
'''Wizualizacja wyników -> scatter rysuje wykres punktowy:
    -> x i y reprezentują wartości kolumny 6 i 7.
    -> c=clusters koloruje punkty na podstawie przypisanych klastrów.
    -> cmap="viridis" używa palety kolorów "viridis" do reprezentacji klastrów.'''

# Opis osi
plt.xlabel(columns[5])
plt.ylabel(columns[6])
plt.title("K-means Clustering of Seeds")

# plt.show()
# plt.savefig("fig_test_t1.png")
plt.savefig("fig_more_clusters_t1.png")

''' Podsumowanie:
- Kod standaryzuje dane, wykonuje klastrowanie za pomocą algorytmu k-średnich
i wizualizuje wyniki w postaci wykresu punktowego z kolorami reprezentującymi klastry.
- Ostateczny wykres przedstawia dwie wybrane kolumny jako osie oraz przypisane klastry jako różne kolory.

KLASTROWANIE -> patrz explanation_file.txt
'''