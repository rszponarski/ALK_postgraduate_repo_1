# TASK 4 MODULE 5 - ADJUSTED RAND SCORE
"""The task is to cluster a dataset of seeds using the k-means method and compare the results with
the actual seed classification given in the Type of seed column.
Evaluate how well the k-means method corresponds to the actual seed classification."""
from sklearn.metrics import adjusted_rand_score
# adjusted_rand_score function: used to evaluate the quality of the fit of the predicted classes to the actual classes.
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Data reading
df = pd.read_csv("seeds_dataset_2.csv")

X = df.drop(["Type of seed"], axis=1)
# – tworzenie X, czyli zbioru cech bez etykiety rzeczywistej (Type of seed).
y_true = df["Type of seed"]
# - wyodrębnienie rzeczywistych etykiet klas (Type of seed), czyli prawdziwe wartości klas
#   (np. klasyfikacja nasion na podstawie ich typu).

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
y_pred = kmeans.fit_predict(X_scaled)
"""– dopasowanie modelu KMeans do skalowanych danych (X_scaled) oraz przypisanie każdego punktu do jednego z 3 klastrów
 (etykiety przypisane do zmiennej y_pred)."""

ari = adjusted_rand_score(y_true, y_pred)
print(f'Adjusted Rand Index: {ari}')

""" NOTES:
--> adjusted_rand_score(y_true, y_pred) – oblicza Skorygowany Wskaźnik Randa, który porównuje
rzeczywiste etykiety (y_true) z przewidywanymi klasami (y_pred).
Wskaźnik ten ocenia jakość dopasowania, biorąc pod uwagę zarówno zgodność, jak i wielkość danych.

--> ari_score to wynik tego porównania, który może być używany do oceny, jak dobrze model przewiduje klasy."""