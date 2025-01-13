# Module 1 Task 7
"""Napisz funkcję load_many(fpaths, add_random_rows=0), która przyjmie listę ścieżek do plików CSV (argument fpaths)
i korzystając z funkcji napisanej w zadaniu „Wczytywanie danych w formacie (x)SV”, wczyta je oraz zwróci jeden df
z ich zawartością."""
import numpy as np
import pandas as pd

paths = ["./dane1.csv", "./dane2.csv", "./dane3.csv"]

# funkcja dummy_data() z zadania 1 - troche inaczej
def dummy_data(rows: int, columns: list, do_nan: bool = False) -> pd.DataFrame:
    if do_nan:
        # Tworzenie tablicy wypełnionej np.nan
        data = np.full((rows, len(columns)), np.nan)
    else:
        # Losowe liczby z przedziału [0, 1]
        data = np.random.rand(rows, len(columns))

    return pd.DataFrame(data, columns=columns)


# funkcja load() z zadania 3- inaczej
def load(fpath: str) -> pd.DataFrame:
    # Wykryj separator wiedząc, że pierwsza kolumna zawsze zawiera int
    with open(fpath, "r") as f:
        _, second_line = f.readline(), f.readline().strip()

    """ Wczytuje plik w formacie (x)SV i automatycznie wykrywa separator.

    # Otwiera plik w trybie odczytu:    with open(fpath, "r") as f:
    # Odczytuje pierwszą linię (nagłówki) i drugą linię (pierwszy wiersz danych)
        _, second_line = f.readline(), f.readline().strip()
    
    _ : używa się do oznaczenia zmiennej, której wartość jest ignorowana (nazywana "dummy variable").
    W tym przypadku oznacza, że pierwsza linia z pliku jest odczytana, ale nie zostaje nigdzie użyta ani zapisana.
    .strip(): Usuwa wszystkie białe znaki z początku i końca odczytanej linii (np. spacje, tab, znaki nowej linii). """

    # Znajdź pierwszy znak, który nie jest cyfrą
    separator = None
    for char in second_line:
        if not char.isdigit() and char not in [' ', '.', '-']:
            separator = char
            break
    """ warunek not char.isdigit() and char not in [' ', '.', '-'] sprawdza, czy znak nie jest ani cyfrą ani żadnym
    z wymienionych znaków specjalnych.
    --> Jeśli warunek if zostanie spełniony, oznacza to, że znaleziono znak, który nie jest cyfrą ani jednym z
    dozwolonych znaków.
    --> Zapisuje ten znak do zmiennej separator.
    --> break: Kończy pętlę for, gdy tylko zostanie znaleziony pierwszy potencjalny separator."""

    # Wczytaj dane
    if separator is None:
        raise ValueError("Nie znaleziono separatora")
    else:
        return pd.read_csv(fpath, sep=separator)


def load_many(fpaths: list, add_random_rows: int = 0) -> pd.DataFrame:
    # Jeśli lista plików jest pusta, zwróć pusty DataFrame
    if not fpaths:  #     if not fpaths oznacza: „jeśli lista fpaths jest pusta”.
        """ sprawdzenie czy użytkownik nie przekazał żadnych plików do przetworzenia.
        Jeśli fpaths jest puste, funkcja natychmiast kończy działanie, zwracając pusty obiekt pd.DataFrame()
        (czyli DataFrame bez kolumn i wierszy)."""
        return pd.DataFrame()

    # Wczytaj wszystkie pliki do listy DataFrame
    dataframes = [load(fpath) for fpath in fpaths]
    """list comprehension: pętla, która iteruje po każdym pliku z listy fpaths"""

    # Połącz wszystkie DataFrame w jeden
    combined_df = pd.concat(dataframes, ignore_index=True)

    # jeśli wymagane, dodaj losowe wiersze
    if add_random_rows > 0:
        """jeśli add_random_rows > 0, oznacza to, że użytkownik chce dodać do wynikowego DataFrame określoną liczbę
        losowych wierszy."""
        # generowanie losowych danych na podstawie kolumn w combined_df
        random_rows = dummy_data(add_random_rows, combined_df.columns.tolist())
        """combined_df.columns.tolist():    Lista nazw kolumn w wynikowym DataFrame (combined_df).
          Wygenerowane dane będą miały:
        --Liczbę wierszy równą add_random_rows.
        --Kolumny odpowiadające nazwom kolumn w combined_df."""
        combined_df = pd.concat([combined_df, random_rows], ignore_index=True)
    return combined_df

# Test
paths = ["./dane1.csv", "./dane2.csv", "./dane3.csv"]
result = load_many(paths, add_random_rows=5)
print(result)

"""PODSUMOWANIE:
Kod ma na celu wczytanie danych z wielu plików CSV/TSV, połączenie ich w jeden zbiorczy DataFrame i opcjonalne dodanie
do niego określonej liczby losowych wierszy. Sposob dzialania:

    1. Sprawdza, czy podano pliki do wczytania:
        Jeśli lista plików (fpaths) jest pusta, zwraca pusty DataFrame.

    2. Wczytuje dane z plików:
        Dla każdego pliku na liście wczytuje jego zawartość jako DataFrame, automatycznie wykrywając separator
        (np. przecinek, tabulator).

    3. Łączy dane:
        Wszystkie wczytane DataFrame'y są łączone w jeden.

    4. Opcjonalnie dodaje losowe wiersze:
        Jeśli użytkownik poda liczbę add_random_rows większą od 0, generuje tyle losowych wierszy, ile zażądano,
        i dodaje je do wynikowego DataFrame.

    5. Zwraca wynik:
        Wynikowy DataFrame zawiera dane z wszystkich plików oraz ewentualnie dodatkowe losowe wiersze.
"""
