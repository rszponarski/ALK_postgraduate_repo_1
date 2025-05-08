# Module 2 Task 5
"""
- Co jesli zbiór danych jest zagnieżdżony jeszcze głębiej?
Napisz funkcję fully_flatten(df), która wykorzysta flatten_column(df, column) z zadania „Incepcja” i będzie spłaszczać
DataFrame tak długo, aż nie znajdzie już więcej żadnych słowników oraz wszystko na pewno będzie dwuwymiarowe.
"""
import pandas as pd


# funkcja z zadania 3
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

    # Dopasuj indeksy do oryginalnego DataFrame !!!! ważne
    # Inaczej będzie cała masa niespodziewanych NaNów, bo konkatenacja się zwali
    flattened = flattened.reindex(df.index, fill_value=None)

    # Usuń oryginalną kolumnę i dołącz spłaszczone dane
    df = df.drop(columns=[column])
    df = pd.concat([df, flattened], axis=1)
    return df


def fully_flatten(df: pd.DataFrame) -> pd.DataFrame:
    """
    Spłaszcza wszystkie zagnieżdżone kolumny w DataFrame.
    """
    while True:
        """
        Pętla jest używana, aby powtarzać proces spłaszczania tak długo, jak w df są kolumny zawierające
        zagnieżdżone struktury (np. słowniki).
        """
        # Znajdź wszystkie kolumny zawierające słowniki (czyli struktury json)
        nested_columns = [
            column for column in df.columns
            if df[column].dropna().apply(lambda x: isinstance(x, dict)).any()
        ]

        # Jeśli brak zagnieżdżonych kolumn, zakończ pętlę
        if not nested_columns:
            """
            Warunek zakończenia pętli:
            Jeśli lista nested_columns jest pusta, to znaczy, że w DataFrame nie ma już zagnieżdżonych struktur.
            W takim przypadku kończymy pętlę.
            """
            break

        # Spłaszczaj każdą zagnieżdżoną kolumne
        for column in nested_columns:
            df = flatten_column(df, column)
            """
            Dla każdej zagnieżdżonej kolumny wywoływana jest funkcja flatten_column, która spłaszcza ją o jeden poziom.
            Wynikowy DataFrame (df) jest na bieżąco aktualizowany.
            """
    return df


if __name__ == "__main__":
    beer_df = pd.read_json("beer.jsonl", lines=True)
