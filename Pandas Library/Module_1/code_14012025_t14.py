# Module 1 Task 14
"""
Napisz funkcję convert_columns_to_datetime(df, date_columns), która przyjmie pd.DataFrame i listę z nazwami kolumn,
które zawierają daty. Funkcja powinna zwrócić zmodyfikowany DataFrame, w którym kolumny z datami zostały
przekonwertowane na typ datetime.

Następnie napisz funkcję time_diff(df, col1, col2), która przyjmie pd.DataFrame wypełniony danymi typu datetime
i nazwy dwóch kolumn z datami, po czym zwróci różnicę czasu między nimi w postaci pd.Timedelta mierzoną w sekundach."""

import pandas as pd


def convert_columns_to_datetime(df: pd.DataFrame, date_columns: list) -> pd.DataFrame:
    """
    Konwertuje kolumny z podanej listy na typ datetime w przekazanym DataFrame.

    Parametry:
    ----------
    df : pd.DataFrame
        DataFrame do modyfikacji.
    date_columns : list
        Lista nazw kolumn do konwersji na datetime.

    Zwraca:
    -------
    pd.DataFrame
        Zmodyfikowany DataFrame z kolumnami typu datetime.
    """
    for col in date_columns:
        df[col] = pd.to_datetime(df[col])  # Konwersja kolumny na datetime
    return df
"""
* Funkcja iteruje i konwertuje wszystkie kolumny z listy date_columns na format datetime64.
* Oznacza to, że każda wartość w tych kolumnach zostaje przekształcona z tekstowej reprezentacji daty na obiekt
datetime Pandas, co umożliwia łatwe operacje takie jak obliczanie różnic, filtrowanie czy grupowanie według dat.
* Zwracamy ten sam DataFrame,
"""

def time_diff(df: pd.DataFrame, col1: str, col2: str) -> pd.Series:
    """
    Oblicza różnicę czasu w sekundach między dwoma kolumnami dat.

    Parametry:
    ----------
    df : pd.DataFrame
        DataFrame zawierający kolumny dat.
    col1 : str
        Nazwa pierwszej kolumny z datami.
    col2 : str
        Nazwa drugiej kolumny z datami.

    Zwraca:
    -------
    pd.Series
        Różnica czasu między dwoma kolumnami w sekundach.
    """
    time_difference = (df[col1] - df[col2]).dt.total_seconds()
    return time_difference
"""
--> (df[col1] - df[col2]): Obliczamy różnicę czasu między wartościami w kolumnach.
    Wynikiem jest Timedelta, czyli obiekt Pandas zawierający różnicę w dniach, godzinach, minutach, itp.
--> .dt.total_seconds():   Konwertujemy Timedelta na liczbę sekund.

* Możesz użyć różnych jednostek, takich jak:
    Days:           diff.dt.days
    Hours:          diff.dt.total_seconds() / 3600
    Minutes:        diff.dt.total_seconds() / 60
    Milliseconds:   diff.dt.total_seconds() * 1000
"""


if __name__ == "__main__":
    # Wczytanie danych z pliku dates.csv
    data = pd.read_csv("./dates.csv")

    # Pobranie listy nazw kolumn
    columns = list(data.columns)

    # Konwersja kolumn na datetime
    modified_data = convert_columns_to_datetime(data, columns)

    # Obliczenie różnicy czasu w sekundach między pierwszą a drugą kolumną
    diff_in_seconds = time_diff(modified_data, columns[0], columns[1])

    # Wyświetlenie różnicy
    print("Różnica czasu (w sekundach):")
    print(diff_in_seconds)
