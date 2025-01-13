import numpy as np
import pandas as pd

# przykładowe dane
n = 5
columns = ['$$$', 2, 'b', 'bla', 5]
nan = False

# np.full((wiersze, kolumny), tym_wypelnij_wszystko)
data1 = np.full((n, len(columns)), 'xyz')
# tworzy tablicę o wymiarach (n, len(columns)) wypełnioną wartością 'xyz' lub np.nan.
print(data1)
print(type(data1))
# tworzenie dataframe
df = pd.DataFrame(data1, columns=columns)
print(f'\n{df}')
print(type(df))

# co gdy podam mniej kolumn niz mam wygenerowanych danych? - WYNIKIEM JEST ERROR
columns2 = ['a', 'b', 'c']  # z data1 wiemy ze mamy tabele 5x5, tutaj podajemy 3 kolumny
# df = pd.DataFrame(data1, columns=columns2)  # to daje blad
'''ValueError: Shape of passed values is (5, 5), indices imply (5, 3)
Błąd oznacza, że próbujesz utworzyć obiekt DataFrame o niezgodnych wymiarach między danymi (values) a listą kolumn
(columns). Gdy próbujesz utworzyć DataFrame, Pandas oczekuje, że liczba kolumn w tablicy (5) będzie zgodna z liczbą
nazw kolumn (3).

Rozwiazanie:
1. Dopasowanie liczby kolumn do tablicy --> dodaj nazwy pozostalych kolumn
2. Zredukowanie tablicy NumPy do wymaganej liczby kolumn --> zrob wycinek swojej tablicy redukujac do do 5x3'''
# Wycinek tablicy ograniczony do 3 kolumn
data_reduced = data1[:, :3]
df = pd.DataFrame(data_reduced, columns=columns2)
print(f'\n{df}')

# np.random.rand((wiersze, kolumny)
data2 = np.random.rand(n, len(columns))
# tworzy tablicę o wymiarach (n, len(columns)) wypełnioną losowymi wartosciami [0.0-1.0]
print(f'\n{data2}')
