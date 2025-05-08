# Module 2 Task 4
"""
Napisz funkcję flatten_nested_columns(df) korzystając ze swojej funkcji z poprzedniego zadania.
Zastosuj ją do wszystkich zagnieżdżonych kolumn w DataFrame'ie.

Wykorzystaj metodę .apply(), która stosuje wybraną funkcję do każdej wartości pd.Series.
Wykorzystaj operator lambda.
"""
import pandas as pd


def flatten_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Spłaszcza kolumnę DataFrame'u o jeden poziom za pomocą json_normalize,
    zachowując poprawne indeksowanie.
    """
    if column not in df.columns:
        raise ValueError(f"Kolumna '{column}' nie istnieje w DataFrame.")

    # Spłaszcz kolumnę zagnieżdżoną (ignoruąc NaN)
    flattened = pd.json_normalize(df[column].dropna())

    # Dodaj prefiks do nazw kolumn
    flattened.columns = [f"{column}_{col}" for col in flattened.columns]

    # Dopasuj indeksy do oryginalnego DataFrame
    flattened = flattened.reindex(df.index, fill_value=None)

    # Usuń oryginalną kolumnę i dołącz spłaszczone dane
    df = df.drop(columns=[column])
    df = pd.concat([df, flattened], axis=1)
    return df


def flatten_nested_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Spłaszcza wszystkie zagnieżdżone kolumny w DataFrame'ie, korzystając z .apply().
    """
    # Znajdź kolumny o typie "object", które mogą zawierać zagnieżdżone struktury
    nested_columns = [col for col in df.columns if df[col].dtype == 'object']
    """
    Tworzymy listę nested_columns, która zawiera nazwy kolumn o typie object.
    W pandas obiekty JSON (np. słowniki) są reprezentowane jako typ object.
    --- Dlaczego? --->
    Tylko kolumny typu object mogą zawierać zagnieżdżone struktury, takie jak słowniki.
    """

    # Dla każdej kolumny sprawdzamy czy zawiera słowniki, jeśli tak, spłaszczamy ją
    for column in nested_columns:
        """
        Iteracja przez potencjalnie zagnieżdżone kolumny
        -> Pętla iteruje przez każdą kolumnę z listy nested_columns.
        --- Dlaczego? --->
        Dla każdej kolumny (z typem object) musimy sprawdzić, czy zawiera zagnieżdżone dane.
        """
        if df[column].dropna().apply(lambda x: isinstance(x, dict)).any():  # if True / False
            # tzn. jesli przynajmniej jeden element w kolumnie column jest typu dict (zagnieżdżoną strukturą danych),
            # funkcja spłaszczy tę kolumnę / Jezeli zaden element nie jest dict, mamy 'if False'; funkcja nic nie zrobi
            # z ta kolumna.
            """
            Sprawdzenie, czy kolumna zawiera słowniki
                * df[column].dropna():
            Usuwamy puste wartości (NaN) z kolumny, bo nie są istotne w tym kroku.
                * .apply(lambda x: isinstance(x, dict)):   <--- SEE NOTE
            Dla każdej wartości w kolumnie sprawdzamy, czy jest to słownik (dict).
            Funkcja isinstance(x, dict) zwraca True dla wartości będących słownikami.
                * .any():
            Sprawdzamy, czy przynajmniej jedna wartość w kolumnie jest słownikiem.
            --- Dlaczego? --->
            Jeśli w kolumnie znajdują się słowniki, oznacza to, że wymaga ona spłaszczenia.
            """
            df = flatten_column(df, column)
            """
            Wywołujemy funkcję flatten_column, aby spłaszczyć zagnieżdżoną kolumnę.
            Wynikowa tabela DataFrame (df) zostaje zaktualizowana o nowe, spłaszczone kolumny.
            """
    return df


if __name__ == "__main__":
    beer_df = pd.read_json("beer.jsonl", lines=True)

"""
NOTES:
Cel funkcji:
Funkcja ma za zadanie spłaszczyć wszystkie zagnieżdżone kolumny w DataFrame, czyli takie, które zawierają zagnieżdżone
obiekty JSON (np. słowniki). Proces ten upraszcza dane, aby mogły być łatwiej analizowane i przekształcane.
---------------------------------
Czym jest lambda?
    --> lambda to sposób definiowania funkcji anonimowej w Pythonie, czyli funkcji, która nie ma nazwy.
        Służy głównie do tworzenia krótkich, jednorazowych funkcji, które są używane w miejscu,
        gdzie ich definicja jest potrzebna.

Składnia:   lambda arguments: expression

    arguments: Lista argumentów, tak jak w funkcji zwykłej.
    expression: Wyrażenie, które zostanie obliczone i zwrócone.
    
            Lambda zawsze zwraca wynik tego wyrażenia.
---------------------------------
Przykład prostej funkcji lambda
Lambda to anonimowa funkcja, którą możesz zdefiniować w jednej linijce. Oto kilka prostych przykładów:

1. Dodawanie dwóch liczb

    add = lambda x, y: x + y
    print(add(3, 5))  # Wynik: 8

2. Podnoszenie liczby do kwadratu

    square = lambda x: x ** 2
    print(square(4))  # Wynik: 16
---------------------------------
Jak działa lambda w tym fragmencie?     df[column].dropna().apply(lambda x: isinstance(x, dict)).any()

Rozbicie na części:
    df[column]:
        -- Pobiera daną kolumnę z DataFrame jako obiekt pd.Series.
    dropna():
        -- Usuwa wartości NaN z serii. Dzięki temu operujemy tylko na rzeczywistych danych.
    apply(lambda x: isinstance(x, dict)):
        -- Funkcja apply() stosuje podaną funkcję (tutaj lambda) do każdej wartości w serii.

        lambda x: isinstance(x, dict):
            -- To anonimowa funkcja, która przyjmuje jedną wartość x jako argument.
            -- Sprawdza, czy x jest słownikiem (dict), za pomocą funkcji isinstance.
            -- Zwraca True, jeśli wartość x jest słownikiem, lub False, jeśli nie.

    Wynikiem tej operacji jest seria (pd.Series) zawierająca wartości True lub False dla każdej pozycji w kolumnie.
"""