# Module 1 Task 12
"""  Napisz funkcję find_books_with_positive_reviews(df), która znajdzie wszystkie książki z przewagą ocen pozytywnych
nad negatywnymi (przyjmij, że oceny negatywne to 1 i 2 gwiazdki zaś oceny pozytywne to 4 i 5 gwizdek).
Zignoruj oceny neutralne (3 gwiazdki).

Następnie napisz drugą funkcję compare_average_price(positive_df, all_df), która porówna średnią cenę książek z przewagą
ocen pozytywnych ze średnią ceną wszystkich książek. Funkcja powinna zwrócić wartość True, jeśli średnia cena książek
z przewagą ocen pozytywnych jest wyższa. Informacja o cenie książki znajduje się w kolumnie price. """

import pandas as pd


def find_books_with_positive_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Znajduje wszystkie książki, w których liczba ocen pozytywnych (ratings_4 i ratings_5)
    przewyższa liczbę ocen negatywnych (ratings_1 i ratings_2), ignorując oceny neutralne.

    In:
    ----------
    df : pd.DataFrame
        Wejściowy DataFrame zawierający kolumny z ocenami: `ratings_1`, `ratings_2`,
        `ratings_4`, `ratings_5`, oraz kolumnę `title` z tytułem książki.

    Out:
    -------
    pd.DataFrame
        Książki, w których liczba ocen pozytywnych jest większa
        niż liczba ocen negatywnych.
    """
    # dodajemy dwie nowe kolumny -> z suma ocen pozytywnych i negatywnych
    # ocena "3" ignorowana
    df['positive'] = df['ratings_4'] + df['ratings_5']
    df['negative'] = df['ratings_1'] + df['ratings_2']

    # FUNKCJA zwraza tylko te książki, gdzie pozytywne oceny są większe od negatywnych
    # FUNKCJA zwraca nowy df:
    return df[df['positive'] > df['negative']]


def compare_average_price(positive_df: pd.DataFrame, all_df: pd.DataFrame) -> bool:
    """
    Porównuje średnią cenę książek z przewagą ocen pozytywnych (z `positive_df`)
    ze średnią ceną wszystkich książek (z `all_df`).

    Parametry:
    ----------
    positive_df : pd.DataFrame
        DataFrame zawierający tylko książki z przewagą ocen pozytywnych.

    all_df : pd.DataFrame
        DataFrame zawierający wszystkie książki.

    Zwraca:
    -------
    bool
        True, jeśli średnia cena książek z przewagą ocen pozytywnych jest wyższa,
        w przeciwnym wypadku False.
    """
    # obliczamy średnią cenę dla książek z przewagą ocen pozytywnych
    avg_positive_price = positive_df['price'].mean()
    # obliczamy średnią cenę dla wszystkich książek
    avg_all_price = all_df['price'].mean()

    return avg_positive_price > avg_all_price


if __name__ == "__main__":
    books_df = pd.read_csv("./books_with_price.csv")
    positive_books_df = find_books_with_positive_reviews(books_df)
    print(positive_books_df.head())
    print()
    result = compare_average_price(positive_books_df, books_df)
    print(result)

"""
PODSUMOWANIE:

Funkcja compare_average_price przyjmuje dwa DataFrame jako argumenty:
    positive_df: DataFrame zawierający książki z przewagą ocen pozytywnych.
    all_df: DataFrame zawierający wszystkie książki.
Funkcja oblicza średnią cenę książek z positive_df i porównuje ją ze średnią ceną wszystkich książek z all_df.
    Na koniec zwraca:
        True, jeśli średnia cena książek z przewagą ocen pozytywnych jest wyższa.
        False, jeśli średnia cena jest niższa lub równa.
        
---------
ALE zeby z niej skorzystac, konieczne bylo stworzenie nowego DF z ocenami pozytywnymi, bazujac na pierwszej funkcji.

Funkcja find_books_with_positive_reviews(df) zwraca DataFrame zawierający tylko te książki,
które mają przewagę ocen pozytywnych nad negatywnymi (ignorując oceny neutralne — 3 gwiazdki).
"""