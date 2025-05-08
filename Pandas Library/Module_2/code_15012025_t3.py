# Module 2 Task 1
"""
 Rozglądając się po bazie danych beer.jsonl, trudno nie zauważyć jednego problemu:
    baza nie jest dwuwymiarowa. Wiele obiektów zawiera obiekty, które zawierają obiekty.
Aby skorzystać z zalet DataFrame'ów, musimy być w stanie przedstawić dane w postaci zwykłej tabeli 2D.

Napisz funkcję flatten_column(df, column), która spłaszczy kolumnę DataFrame'u o jeden poziom.
Funkcja powinna przyjąć kolumnę, zinterpretować ją jako obiekt JSON i spłaszczyć ją, dodając nowe kolumny do df'u.
Wartości w nowych kolumnach powinny być wartościami zagnieżdżonych obiektów.
Pamiętaj, aby dopasować indeksy do oryginalnego DataFrame. Zastosuj swoją funkcję do kolumny "glass".

Przykład:

Dla kolumny "glass" i jej podkolumn "name", "createDate"
Twoja funkcja powinna stworzyć kolumny o nazwach: "glass_name" i "glass_createDate".
"""
import pandas as pd


def flatten_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    # pd.json_normalize() do spłaszczenia zagnieżdżonej kolumny
    flattened = pd.json_normalize(df[column])
    """
    Funkcja pd.json_normalize() przekształca zagnieżdżoną strukturę JSON w tabelaryczną, gdzie każda podwłaściwość
    obiektu staje się osobną kolumną.
    Wynik to nowy DataFrame, który zawiera spłaszczone dane z kolumny column. <-- SEE FILE t3_normalized.py
    """

    # trzeba dodac prefiks do nazw nowych kolumn, aby były jednoznaczne
    flattened.columns = [f"{column}_{subcol}" for subcol in flattened.columns]
    """
    Aby uniknąć konfliktu nazw, dodajemy prefiks (np. glass_) do każdej nowej kolumny wynikowej.
    Dzięki temu wiemy, że nowe kolumny pochodzą ze spłaszczonej kolumny glass.
    """

    # dopasowanie indeksow nowego df'u do oryginalnego
    flattened.index = df.index
    """
    Indeksy spłaszczonego DataFrame muszą być zgodne z indeksami oryginalnego DataFrame,
    aby po połączeniu wiersze były poprawnie sparowane.
    """

    # polaczenie oryginalnego df ze spłaszczonymi danymi
    df = pd.concat([df.drop(columns=[column]), flattened], axis=1)
    """
    Funkcja zwraca zmodyfikowany DataFrame z rozdzielonymi i spłaszczonymi danymi.
    """
    return df


if __name__ == "__main__":
    # Wczytanie danych z pliku beer.jsonl
    beer_df = pd.read_json("beer.jsonl", lines=True)

    # Nazwa kolumny do spłaszczenia
    col = "glass"

    # Spłaszczanie kolumny
    beer_df = flatten_column(beer_df, col)

    print(beer_df.head())

