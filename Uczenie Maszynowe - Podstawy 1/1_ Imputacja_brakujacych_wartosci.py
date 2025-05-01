""""
ZADANIE 1/1
Wczytaj dane houses.csv i usuń brakujące wartości przy pomocy SimpleImputer.
Dane numeryczne powinny być zastąpione strategią "median", kategoryczne - "most frequent".
Dane z wypełnionymi brakami powinny być zapisane w ramce danych "houses". """

import pandas as pd
from sklearn.impute import SimpleImputer # -> SimpleImputer z sklearn.impute pozwala na automatyczne uzupełnianie
                                            # brakujących danych według wybranej strategii.

# Wczytaj dane
df = pd.read_csv('houses.csv')

# Podział na dane numeryczne i kategoryczne
num_cols = df.select_dtypes(include=['int64', 'float64']).columns  # kolumny z liczbami (int i float)
cat_cols = df.select_dtypes(include=['object', 'category']).columns  # kolumny z tekstami (typ object lub category), czyli dane kategoryczne
'''select_dtypes() ---> pozwala wybrać kolumny konkretnego typu danych.'''

# Imputacja danych numerycznych strategią 'median'
num_imputer = SimpleImputer(strategy='median')
df[num_cols] = num_imputer.fit_transform(df[num_cols])
'''Tworzymy obiekt SimpleImputer, który będzie wypełniał brakujące liczby medianą kolumny -> kolumny numeryczne
- fit_transform() uczy się mediany z danych i w tym samym kroku je wypełnia.
- Wynik podstawiamy z powrotem do tych samych kolumn w df. '''

# Imputacja danych kategorycznych strategią 'most_frequent'
cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
'''to samo co wcześniej, ale kolumny kategoryczne uzupełniamy 'most frequent' zamiast mediany'''

# Gotowa ramka danych po imputacji
houses = df
'''nadpisywalismy wartości kolumn w miejscu więc tylko trzeba stworzyc nową ramkę danych'''

# wyświetl kilka pierwszych wierszy
print(houses.head())

# zapis do nowego pliku csv
houses.to_csv('houses_clean.csv', index=False)