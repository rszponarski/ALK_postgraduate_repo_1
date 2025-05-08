# Module 1 Task 11
""" plik books.csv zawiera między innymi kolumny (ratings_1, ..., ratings_5) z ocenami książek.
Sufiksy nazw kolumn oznaczają odpowiednio od 1 do 5 przyznanych gwiazdek.
W kolumnach natomiast znajdują się liczby ocen o danej ilości gwiazdek.

Napisz funkcję add_rating_percentages(df), która stworzy pięć nowych kolumn – percent_1, ..., percent_5 –
zawierających informacje o tym, jaką część wszystkich ocen tej książki stanowią oceny o danej liczbie gwiazdek
oraz zwróci zmodyfikowany DataFrame z dodatkowymi kolumnami. """

import pandas as pd


def add_rating_percentages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Modyfikuje przekazany DataFrame, dodając nowe kolumny `percent_1` do `percent_5`.
    Każda z tych kolumn reprezentuje procentowy udział danej oceny (1-5 gwiazdek)
    w całkowitej liczbie ocen dla każdej książki.

    In:
    ----------
    df : pd.DataFrame
        Wejściowy DataFrame zawierający kolumny z ocenami: `ratings_1` do `ratings_5`.

    Out:
    -------
    pd.DataFrame
        Ten sam obiekt DataFrame z dodanymi pięcioma nowymi kolumnami.
        Uwaga: Funkcja modyfikuje DataFrame wejściowy bez tworzenia jego kopii.

    Modyfikacje:
    ------------
    - Dodaje kolumny:
        `percent_1` - Procentowy udział ocen 1-gwiazdkowych.
        `percent_2` - Procentowy udział ocen 2-gwiazdkowych.
        `percent_3` - Procentowy udział ocen 3-gwiazdkowych.
        `percent_4` - Procentowy udział ocen 4-gwiazdkowych.
        `percent_5` - Procentowy udział ocen 5-gwiazdkowych.
    """
    # suma wszystkich ocen dla danej książki
    total_ratings = df[['ratings_1', 'ratings_2', 'ratings_3', 'ratings_4', 'ratings_5']].sum(axis=1)
                                                    # axis=1 oznacza sumowanie wierszy wskazanych kolumn.

    # dodanie nowych kolumn do tego samego df
    df['percent_1'] = (df['ratings_1'] / total_ratings)  # * 100 gdybysmy chcieli w procentach
    df['percent_2'] = (df['ratings_2'] / total_ratings)  # * 100
    df['percent_3'] = (df['ratings_3'] / total_ratings)  # * 100
    df['percent_4'] = (df['ratings_4'] / total_ratings)  # * 100
    df['percent_5'] = (df['ratings_5'] / total_ratings)  # * 100

    return df


if __name__ == "__main__":
    books_df = pd.read_csv("./books.csv")
    print(books_df.columns)
    books_df = add_rating_percentages(books_df)
    print(books_df.columns, "\n")
    print(books_df.head(2))
    """ Aby wyświetlić tylko ostatnie 5 kolumn w DataFrame, możesz użyć następującej konstrukcji:
        df.iloc[:, -5:] """
    print(books_df.iloc[:, -5:].head(2), "\n")