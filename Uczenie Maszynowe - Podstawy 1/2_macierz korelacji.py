""""
ZADANIE 1/2
Stwórz plik graficzny z przygotowaną przez siebie macierzą korelacji na danych houses.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Przykład obliczania korelacji
df = pd.read_csv('houses_clean.csv')
num_cols = df.select_dtypes(include=['int64', 'float64']).columns  # wybieramy tylko kolumny numeryczne
correlation_matrix = df[num_cols].corr()
'''df[num_cols] – wyodrębniamy tylko numeryczne kolumny z ramki danych.
   .corr() – oblicza macierz korelacji między tymi kolumnami.
Metoda ta zwraca tabelę, w której każda komórka odpowiada za korelację między dwiema kolumnami.'''

# Można po prostu wyświetlić macierz
print(correlation_matrix)

'''ALE zależy nam na wizualizacji'''

# Tworzenie wykresu
plt.figure(figsize=(10, 8))  # rozmiar wykresu na 10x8 cali
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Macierz korelacji')
'''funkcja z biblioteki seaborn rysuje heatmapę macierzy korelacji. Używamy:
    annot=True – aby wartości korelacji były wyświetlane na wykresie.
    cmap='coolwarm' – wybór kolorystyki wykresu, gdzie czerwony to silna korelacja dodatnia, a niebieski silna korelacja ujemna.
    fmt='.2f' – liczby po przecinku w formacie 2 miejsc po przecinku.
    linewidths=0.5 – ustawia grubość linii między komórkami wykresu.'''

# Zapisanie wykresu do pliku
plt.savefig('macierz_korelacji.png', dpi=300)

"""
NOTATNIK:
Co to jest macierz korelacji?
---> to narzędzie statystyczne, które pozwala na ocenę wzajemnych zależności między ZMIENNYMI NUMERYCZNYMI
w zbiorze danych. Korelacja mierzy, jak silnie dwie zmienne są ze sobą powiązane. Korelacja może być:

    Dodatnia (pozytywna): Kiedy jedna zmienna rośnie, druga również rośnie (np. wzrost wzrostu a waga).
    Ujemna (negatywna): Kiedy jedna zmienna rośnie, druga maleje.
    Brak korelacji: Kiedy zmienne nie mają między sobą żadnej zależności (np. wzrost a liczba posiłków dziennie).

- Korelacja pozwala zrozumieć, które zmienne mają ze sobą silne zależności i które mogą być ze sobą powiązane w sposób
przyczynowo-skutkowy.

- Jeśli pracujesz nad modelem uczenia maszynowego, macierz korelacji pomoże Ci zrozumieć, które zmienne
są podobne i mogą wprowadzać redundancję.
Może to pomóc w wyborze najbardziej istotnych cech i unikaniu KOLINEARNOŚCI (czyli silnej korelacji między zmiennymi).

- Silne korelacje mogą wskazywać na błąd w danych, np. dwa identyczne atrybuty mogą sugerować duplikaty.
- Wybór algorytmów: Modele uczenia maszynowego mogą różnie radzić sobie z korelacjami między zmiennymi.
Zrozumienie zależności może pomóc w wyborze najlepszych metod (np. regresja liniowa, drzewa decyzyjne, itp.).
"""