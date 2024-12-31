# TASK 4 MODULE 4

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Insert your code here
df = pd.read_csv('pizza.csv').drop(['id', 'brand'], axis=1)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

'''Określenie liczby głównych składowych do 3 oznacza, że interesuje nas uproszczenie danych do trzech wymiarów.
Wybór 3 wynika z wcześniejszej analizy proporcji wyjaśnianej wariancji — być może 3 składowe wystarczają, by zachować
większość informacji zawartej w danych.
Ze wcześniejszego zadania -> trzy pierwsze składowe wyjaśniały 59,6% + 32,7% + 5,92% = 98,2% wariancji,
co może być wystarczające do analizy.'''
pca = PCA(n_components=3)
''' Użycie fit_transform dla trzech składowych:
fit_transform wylicza wartości głównych składowych (PC1, PC2, PC3) i przekształca dane wejściowe do tego nowego,
uproszczonego układu współrzędnych.
Otrzymujemy tablicę 2D o kształcie (liczba_obserwacji, 3) (czyli dane zredukowane do trzech wymiarów).'''
principalComponents = pca.fit_transform(scaled_data)
'''     = principalComponents = pca.fit_transform(scaler.fit_transform(df))
jest poprawny i działa dokładnie tak samo, jak wcześniejszy kod. Można go stosować, ponieważ scaler.fit_transform(df)
zwraca dane znormalizowane, które następnie są przekazywane do pca.fit_transform().

*** Dlaczego pojawia się wrażenie "podwójnego skalowania"? ***
    Wrażenie to może wynikać z niejasności dotyczącej kolejnych operacji wykonywanych na danych,
    ale w rzeczywistości dane są skalowane tylko raz (przez scaler.fit_transform(df)),
    a późniejsze operacje PCA (pca.fit_transform) nie są skalowaniem, ale zupełnie innym procesem.
    
Oto, co dokładnie się dzieje:
    --> scaler.fit_transform(df):
    Skaluje dane tak, aby każda cecha (kolumna) miała średnią 0 i odchylenie standardowe 1.
    To konieczne, aby PCA działało poprawnie, ponieważ PCA jest wrażliwe na różnice w skalach pomiędzy cechami.
    --> pca.fit_transform(scaled_data):
                                            TO NIE JEST SKALOWANIE!
    PCA przekształca dane (już znormalizowane) na nowe wymiary (główne składowe).
    
    Podczas tej operacji PCA:
    -> Oblicza macierz kowariancji na podstawie skalowanych danych.
    -> Znajduje wektory własne i wartości własne tej macierzy.
    -> Przekształca dane, aby wyrazić je w układzie współrzędnych głównych składowych.

Kluczowa różnica między skalowaniem a PCA:
--> Skalowanie (scaler.fit_transform):  Przekształca dane tak, aby każda kolumna miała średnią 0 i odchylenie std;
                                    Żadne relacje pomiędzy cechami nie są zmieniane.

--> PCA (pca.fit_transform):    Redukuje wymiar danych i transformuje je na główne składowe. Główne składowe to nowe
                                zmienne, które reprezentują dane, maksymalizując wariancję w poszczególnych kierunkach.

Dlaczego oba kroki są konieczne?
Skalowanie: PCA opiera się na macierzy kowariancji, która jest zależna od skali cech. Jeśli cechy mają różne skale
            (np. jedna mierzona w milimetrach, a druga w tonach), wynik PCA będzie zdominowany przez te cechy,
            które mają większą skalę. Dlatego wszystkie cechy są sprowadzane do wspólnej skali przed wykonaniem PCA.

PCA: Po znormalizowaniu danych PCA identyfikuje nowe osie (główne składowe), które najlepiej reprezentują zróżnicowanie
 danych. Ten krok przekształca dane w nowy układ współrzędnych.

Czy to jest "podwójne skalowanie"?
Nie. Skalowanie odbywa się tylko raz. Drugi fit_transform (od PCA) nie jest skalowaniem,
                                                            ale przekształceniem danych na główne składowe.

Podsumowanie
scaler.fit_transform(df): --------> Skaluje dane.
pca.fit_transform(scaled_data):  --------> Przekształca już znormalizowane dane na główne składowe.'''

# Visualization
'''Wykres dwóch pierwszych składowych:
Chociaż obliczono 3 składowe, na wykresie przedstawiamy tylko dwie pierwsze (PC1 i PC2), ponieważ:
Wykres dwuwymiarowy jest najłatwiejszy do interpretacji wizualnie.
PC1 i PC2 wyjaśniają najwięcej wariancji, więc zawierają najwięcej informacji o danych.
Trzecia składowa (PC3) jest nadal dostępna, ale jej wpływ może być analizowany w inny sposób, np. w analizie
przestrzennej 3D lub w dalszych etapach analizy.'''
plt.scatter(principalComponents[:, 0], principalComponents[:, 1])
plt.title('PCA')
plt.xlabel('First main component')
plt.ylabel('Second main component')
plt.savefig("fig2.png")
''' array[:, 0]: Wybiera wszystkie wiersze, ale tylko pierwszą kolumnę.'''
