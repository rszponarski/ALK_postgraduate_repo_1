import pandas as pd
import numpy as np

# Tworzenie serii danych
"""
1. Stworzenie slownika:
Kluczami są nazwy kolumn ('A', 'B', 'C', 'D').
Wartościami są listy (lub inne struktury danych), które zawierają wartości odpowiadające poszczególnym kolumnom."""

data = {
    'A': [1.0, np.nan, 4.0, np.nan],
    'B': [np.nan, 2.0, 5.0, np.nan],
    'C': [3.0, 3.0, np.nan, np.nan],
    'D': [np.nan, 3.0, np.nan, np.nan]
}

"""
2. Tworzenie dataframe"""
df = pd.DataFrame(data)

print(df, "\n")
print(df.shape, "\n")   # zwraca tuples w formacie (liczba_wierszy, liczba_kolumn)
print(df.info(), "\n")  # wyświetla szczegółowe informacje, w tym liczbę wierszy, kolumn,
                        # a także dane o typach danych oraz liczbie brakujących wartości (NaN)
"""
3. Obliczenie maksymalnej DOPUSZCZALNEJ liczby brakujących wartości w wierszu"""
# Zakladamy, ze max. moze brakowac 2 z 4 wartosci, wiec 50%
threshold = 0.5
max_missing_allowed = threshold * len(df.columns)
# np. 0.5 * 4 = 2

"""
4. Tworze nowy df, ktory ma usuniete wiersze zgodnie z warunkiem."""
new_df = df[df.isna().sum(axis=1) <= max_missing_allowed]  # <=
new_df2 = df[df.isna().sum(axis=1) < max_missing_allowed]  # <

# SPRAWDZENIE czy usunieto ostatni wiersz:
print(new_df, "\n")
print(new_df.shape,  "\n")
print(new_df2, "\n")

"""
5. Ponowna konwersja df na dict:"""
# Aby przekonwertować DataFrame z powrotem na słownik, można skorzystać z metody to_dict().
cleared_dict = new_df.to_dict()
print(cleared_dict)