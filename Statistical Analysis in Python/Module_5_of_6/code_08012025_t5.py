# TASK 5 MODULE 5

# import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
# TSNE – funkcja z biblioteki sklearn.manifold umożliwiająca redukcję wymiarowości danych z użyciem algorytmu t-SNE."""

# Loading a sample dataset and omitting the id cluster
data = pd.read_csv('pizza.csv').drop('id', axis=1)  # – odczyt danych z pliku CSV oraz usunięcie kolumny 'id'.
X = data.drop('brand', axis=1)  # – przygotowanie cech (wszystkich kolumn oprócz 'brand').
y = data['brand']  # – przygotowanie rzeczywistych etykiet klas (brand), czyli wartości przypisanych do marek pizz.

# Performing the t-SNE transformation
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)
"""TSNE(n_components=2, random_state=42) – inicjalizacja algorytmu t-SNE z dwoma komponentami (n_components=2)
    oraz ustalonym ziarno (random_state=42).
X_tsne = tsne.fit_transform(X) – wykonanie redukcji wymiarowości dla danych X do dwóch wymiarów (X_tsne)."""


# Creation of a new data set containing the 'brand' column
df = pd.DataFrame(X_tsne, columns=['t-SNE 1', 't-SNE 2'])
df['brand'] = y
''' – tworzenie nowego df z dwoma kolumnami: 't-SNE 1' i 't-SNE 2', które zawierają wyniki redukcji wymiarowości.
    – dodanie oryginalnych etykiet marek (y) jako nowej kolumny do DataFrame df.'''

# Data visualisation after dimensionality reduction
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='t-SNE 1', y='t-SNE 2', hue='brand')
plt.title('t-SNE - visualisation with colouring by brand')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.savefig('fig5.png')

""" NOTES:
CEL ZADANIA: Zadanie skupia się na redukcji wymiarowości danych i ich wizualizacji,
                aby odkryć ukryte wzorce i zależności między różnymi markami pizzy."""