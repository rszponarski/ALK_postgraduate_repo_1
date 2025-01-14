# Module 1 Task 13
"""Napisz funkcję merge_dataframes_with_transposition_check(df1, df2), która przyjmie dwa obiekty pd.DataFrame'y,
porówna kolumny biorąc poprawkę na możliwą transpozycję i jeśli są takie same, połączy je w jeden pd.DataFrame.
W przeciwnym wypadku funkcja powinna zwrócić None. """

"""Co oznacza "biorąc poprawkę na możliwą transpozycję"? --> SEE NOTE"""

import pandas as pd


def merge_dataframes_with_transposition_check(df1: pd.DataFrame, df2: pd.DataFrame):
    """
    Łączy dwa DataFrame'y, sprawdzając zgodność kolumn z poprawką na możliwą transpozycję.

    Parametry:
    ----------
    df1 : pd.DataFrame
        Pierwszy DataFrame.
    df2 : pd.DataFrame
        Drugi DataFrame.

    Zwraca:
    -------
    pd.DataFrame lub None
        Połączony DataFrame, jeśli kolumny zgadzają się biorąc pod uwagę transpozycję,
        w przeciwnym razie None.
    """
    # Sprawdź, czy kolumny obu DataFrame są zgodne
    if set(df1.columns) == set(df2.columns):  # tworzy zbiór nazw kolumn z df1 oraz zbiór nazw kolumn z df2.
        return pd.concat([df1, df2], ignore_index=True)
    """
    --> DLACZEGO set() ?
        Użycie set w tym kontekście ułatwia porównanie nazw kolumn lub indeksów w dwóch DataFrame'ach,
        niezależnie od ich kolejności. Wykorzystujemy naturalną cechę zbiorów (set), która sprawia, że:
        * Nie mają kolejności
        * Przechowują tylko unikalne wartości
    --> Porównanie ==: Sprawdza, czy oba zbiory są identyczne, czyli czy obie DataFrame'y mają te same kolumny
                        (kolejność nie ma znaczenia).
    --> pd.concat([df1, df2], ignore_index=True): Łączy dwa DataFrame'y wzdłuż osi wierszy.
    --> ignore_index=True: Tworzy nowe indeksy dla wynikowego DataFrame, ignorując oryginalne indeksy z df1 i df2.
    --> jeśli kolumny są zgodne, funkcja zwraca połączony DataFrame.
    
    Jesli warunek jest prawdziwy: python przerywa dalsze sprawdzanie i wychodzi z funkcji, zwracając wynik. 
        JESLI NIE:
    """
    # Sprawdź, czy kolumny df1 pasują do indeksów df2 (transpozycja df2)
    if set(df1.columns) == set(df2.index):
        df2_transposed = df2.T  # Transpozycja df2
        return pd.concat([df1, df2_transposed], ignore_index=True)
    """
    Jesli warunek jest prawdziwy, tzn. ze jeden df zapisany jest w postaci normalnej, natomiast drugi w transponowanej.
    Robimy transpozycje i nastepnie merge obu df2.
    """

    # Sprawdź, czy indeksy df1 pasują do kolumn df2 (transpozycja df1)
    if set(df1.index) == set(df2.columns):
        df1_transposed = df1.T  # Transpozycja df1
        return pd.concat([df1_transposed, df2], ignore_index=True)
    """ Analogicznie do tego co wczesniej; tyle ze transponujemy df1. """

    # Jeśli żadne z powyższych warunków nie są spełnione, zwróć None
    return None


# TESTY
if __name__ == "__main__":
    books_df = pd.read_csv("./books_1401.csv")
    books_transposed_df = pd.read_csv("./other_books.csv", index_col=0)

    result = merge_dataframes_with_transposition_check(books_df, books_transposed_df)
    if result is not None:
        print("DataFrames zostały połączone:")
        print(result.head())
    else:
        print("DataFrames nie mogą zostać połączone.")

    # RESULT: [5 rows x 22 columns] bo wyswietlamy head() ------------------
    print(result.shape, "\n")
    # SPRAWDZENIE:
    print(books_df.shape, "\n")
    print(books_transposed_df.T.shape)
    print(books_df.shape[0] + books_transposed_df.T.shape[0])  # zgadza sie

""" NOTE:

Transpozycja w kontekście DataFrame'ów oznacza zamianę wierszy z kolumnami.
W praktyce dane w pliku CSV mogą być zapisane w różnych formatach:
    Normalny układ: Każda kolumna to atrybut (np. oceny, ceny), a każdy wiersz to rekord (np. książka).
    Transponowany układ: Każdy wiersz to atrybut, a każda kolumna to rekord.

Przykład:

    Normalny układ:
    title	ratings_1	ratings_2	ratings_3
    Book A	100	        50	        20
    Book B	80	        60	        10

    Transponowany układ:
    	Book A	Book B
    ratings_1	100	80
    ratings_2	50	60
    ratings_3	20	10

Dlaczego trzeba brać pod uwagę transpozycję?
    --Różne formaty danych w plikach
    Plik CSV może być zapisany w jednym z powyższych układów. Jeśli chcemy porównać lub połączyć dwa DataFrame'y,
    musimy uwzględnić możliwość, że jeden z nich moze byc transponowany względem drugiego.

    --Niekompatybilność struktur
    Bez uwzględnienia transpozycji, porównanie kolumn dwóch DataFrame'ów może zakończyć się błędem, mimo że dane są
    identyczne, tylko zapisane w różnych układach.

    --Automatyzacja analizy danych
    Uwzględnianie transpozycji pozwala automatycznie dopasowywać różne układy danych, co czyni funkcję bardziej
    uniwersalną i odporną na błędy formatowania.

Dlaczego transpozycja jest istotna w tym zadaniu?
W podanym zadaniu, jeden z plików (other_books.csv) może być zapisany w formie transponowanej względem books.csv.
Kiedy kolumny df1 pasują do indeksów df2 (lub odwrotnie):
Jeśli dane są transponowane, nazwy kolumn jednego DataFrame mogą odpowiadać indeksom drugiego. W takim przypadku musimy transponować jeden z DataFrame'ów, aby móc je porównać i połączyć.

Przykład:

    Kolumny df1: ['ratings_1', 'ratings_2', 'ratings_3']
    Indeksy df2: ['ratings_1', 'ratings_2', 'ratings_3']
    Transpozycja df2 umożliwi porównanie danych.

Dopasowanie struktur:
Po transpozycji, DataFrame'y mogą mieć identyczne kolumny, co umożliwia ich łączenie.
"""