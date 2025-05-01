# Module 2 Task 8
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# Importuje klasę MinMaxScaler z biblioteki scikit-learn, która pozwala normalizować dane do zakresu [0, 1].

def normalize_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    df:      DataFrame Pandas, który zawiera dane do normalizacji.
    columns: Lista nazw kolumn, które mają być znormalizowane.
    
    scaler:  Tworzy instancję MinMaxScaler jako scaler.
    scaler.fit_transform(df[columns]):
        Normalizuje wskazane kolumny (columns) w DataFrame.
        Wynik jest zwracany jako tablica NumPy, którą przypisujemy do oryginalnych kolumn w DataFrame (df[columns]).
        Funkcja zwraca zaktualizowany DataFrame z znormalizowanymi wartościami w podanych kolumnach.
    """
    scaler = MinMaxScaler()

    # --> stosowanie skali dla wybranych kolumn
    df[columns] = scaler.fit_transform(df[columns])

    return df


if __name__ == "__main__":
    beer_df = pd.read_json("imputed_beer.jsonl", lines=True)
    cols = ["ibu", "abv"]

    beer_df_copy = beer_df.copy()  # <-- trzeba zrobic to samo co w poprzednim zadaniu, aby nie nadpisac beer_df

    # Normalizacja kolumn
    normalized_df = normalize_columns(beer_df_copy, columns=cols)

    # Wyświetl pierwsze wiersze
    # print(normalized_df.head())
    print(beer_df["ibu"].head(10))
    print(normalized_df["ibu"].head(10))

    """
    ->  Odczytuje dane z pliku imputed_beer.jsonl jako DataFrame beer_df.
    ->  Definiuje listę kolumn do normalizacji (cols = ["ibu", "abv"]), które mogą oznaczać np. goryczkę (ibu)
        i zawartość alkoholu (abv).
    ->  Po wykonaniu, kolumny "ibu" i "abv" w beer_df będą przekształcone tak, że ich wartości znajdą się
        w przedziale [0, 1].
    """

"""
NOTES:
- Co to jest normalizacja?

    -> Normalizacja to proces skalowania danych tak, aby wartości znajdowały się w określonym zakresie,
    najczęściej w przedziale [0,1][0,1] lub [−1,1][−1,1].
    
    -> Normalizacja polega na przekształceniu każdej wartości w danych, aby zachowała proporcję względem innych wartości
    w tej samej kolumnie, ale mieściła się w zdefiniowanym zakresie.
    
    -> Formuła najczęściej stosowana do normalizacji (dla przedziału [0,1][0,1]):
    
-----------------------------------------------------------------------------------------------------------------------
- Dlaczego warto normalizować dane?

Normalizacja jest ważna w wielu przypadkach, zwłaszcza w analizie danych i uczeniu maszynowym, z kilku powodów:
    1. Algorytmy wrażliwe na skalę danych:
        Wiele algorytmów uczenia maszynowego (np. regresja logistyczna, k-średnich, sieci neuronowe, SVM) działa lepiej,
        gdy dane mają podobne skale.
        Przykład:
            Jeśli jedna cecha (kolumna) ma wartości w zakresie 1−10001−1000, a inna w 0−10−1, algorytm może nadawać
            nieproporcjonalnie dużą wagę cechom z większymi wartościami, co prowadzi do błędów.
    
    2. Przyspieszenie obliczeń:
        Normalizacja upraszcza obliczenia matematyczne (np. odległości Euklidesowej, która wymaga równych skal cech).
        Znormalizowane dane mogą przyspieszyć zbieżność algorytmów optymalizacyjnych, takich jak gradient prosty.
    
    3. Zapewnienie porównywalności:
        Wartości w różnych kolumnach stają się porównywalne, nawet jeśli mają różne jednostki (np. kilometry i sekundy).
    
    4. Poprawa wizualizacji danych:
        Normalizacja pozwala lepiej wizualizować dane na wykresach, gdzie różnice w skalach mogą zaburzać interpretację.
"""