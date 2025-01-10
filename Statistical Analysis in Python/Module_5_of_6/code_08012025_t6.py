# TASK 6 MODULE 5
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np

# Loading data with the omission of id and brand columns
X = pd.read_csv('pizza.csv').drop(['id', 'brand'], axis=1)

# Performing the t-SNE transformation
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)
'''TSNE(n_components=2): Konfiguracja algorytmu t-SNE -> redukuje wymiary danych do 2D, co umożliwia ich wizualizację
na płaszczyźnie.
random_state=42: Zapewnienie powtarzalności wyników przez ustalenie ziarna generatora liczb losowych.
fit_transform(X): Algorytm uczy się struktury danych X i generuje nowe współrzędne w przestrzeni 2D (X_tsne).
'''

# Selecting a feature to colour - for example, "ash"
feature_to_color = X['ash']
'''Kolumna ash (np. zawartość popiołu w pizzy) wybrana jako cecha, według której będą kolorowane punkty na wykresie.'''

# Drawing a t-SNE diagram with colouring according to the selected characteristic
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=feature_to_color)
plt.colorbar(scatter)
plt.title('t-SNE visualisation with colouring by "ash" feature')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
#plt.savefig('fig6.png')
plt.show()
'''NOTES:
-SNE grupuje punkty, które są do siebie podobne w przestrzeni wielowymiarowej, w taki sposób, aby znajdowały się blisko
siebie na wykresie 2D. Dzięki temu można wizualizować struktury i grupy w danych, które byłyby trudne do dostrzeżenia
w przestrzeni o wysokiej liczbie wymiarów.
'''