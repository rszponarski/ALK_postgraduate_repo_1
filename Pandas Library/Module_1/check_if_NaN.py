import pandas as pd

df = pd.read_csv("./raw_data.csv")
"""
1. Wyświetlenie wszystkich pustych komórek
Aby sprawdzić, które komórki w DataFrame są puste:"""
# Tworzy maskę logiczną wskazującą na puste komórki
empty_cells = df.isna() # isna() w Pandas służy do sprawdzania, które elementy w DataFrame (lub Series) są brakujące
                        # (NaN, czyli Not a Number). Zwraca DataFrame tej samej wielkości co oryginalny DataFrame.
print("Maska pustych komórek:")
print(empty_cells)
# Wyświetla lokalizację pustych komórek
print("Puste komórki (wiersz, kolumna):")
print(empty_cells.stack()[empty_cells.stack()]) # służy do znajdowania lokalizacji brakujących wartości (NaN) w df


""" 2. Wyświetlenie podzbioru z pustymi wartościami
Aby zobaczyć tylko te KOLUMNY, które mają przynajmniej jedną pustą wartość:"""
# Kolumny, które zawierają przynajmniej jedno NaN
columns_with_nans = df.columns[df.isna().any()]
print("Kolumny z pustymi wartościami:", list(columns_with_nans))


""" 3. Wyświetlenie podzbioru z pustymi wartościami
Aby zobaczyć tylko te WIERSZE, które mają przynajmniej jedną pustą wartość:"""
# Wiersze, które zawierają przynajmniej jedno NaN
rows_with_nans = df[df.isna().any(axis=1)]
print("Wiersze z pustymi wartościami:")
print(rows_with_nans)

# ----------------------------------------------------------------------------------------
"""Jak działa mean() w obecności NaN? --> oblicza srednia ignorujac NaN (Not a Number).
Przykład:"""

import pandas as pd
import numpy as np

data = {"A": [6, 2, np.nan, 4]}
df = pd.DataFrame(data)

# Obliczanie średniej
mean_value = df["A"].mean()
print(mean_value)

# Wynik to: (6+2+4) / 3 = 4
# NIE: (6+2+4) / 4
