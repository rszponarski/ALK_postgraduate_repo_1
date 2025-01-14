# Module 1 Task 10
""" Napisz funkcję drop_rows_with_missing_values(df, threshold), która usunie wszystkie wiersze,
    w których brakuje więcej niż threshold * 100% wartości, gdzie threshold to liczba z przedziału [0, 1]."""
import pandas as pd

def drop_rows_with_missing_values(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    FUNKCJA przyjmuje df oraz dopuszczalny prog [0, 1];
    Usuwa wiersze, w których brakujące wartości przekraczają próg określony przez threshold;
    FUNKCJA zwraca df bez wierszy przekraczających próg brakujących wartości.
    """
    # obliczenie maksymalnej DOPUSZCZALNEJ liczby brakujących wartości w wierszu
    max_missing_allowed = threshold * len(df.columns) # <- maksymalnie moze brakowac danych dla wszystkich kolumn
    # np. 0.5 * 10 = 5

    # Filtruje wiersze, w których liczba brakujących wartości nie przekracza max_missing_allowed
    new_df = df[df.isna().sum(axis=1) <= max_missing_allowed] # <-- SEE NOTE
    # np. znalezlismy 3x NaN wsdluz wiersza wiec: 3 <= 5
    # tzn. dodajemy do df[] bo warunek jest TRUE
    return new_df

if __name__ == "__main__":
    test_df = pd.read_csv("./dane1401.csv")
    drop_rows_with_missing_values(test_df, 0.1)

"""NOTES:
    df[df.isna().sum(axis=1) <= max_missing_allowed] 
    
1. df.isna() tworzy maskę logiczną, która identyfikuje brakujące wartości (NaN) w DataFrame.
    True -> jeśli wartość jest NaN
    False -> jeśli wartość nie jest NaN
2. df.isna().sum(axis=1)
    sum(axis=1) sumuje wartości logiczne (True = 1, False = 0) wzdłuż wierszy.
    Oznacza to, że liczymy, ile braków (NaN) występuje w każdym wierszu. = JAK WIELE JEST TRUE CZYLI NaN?
    axis=1 <- wiersze
    axis=0 <- kolumny
3.df[...]
    W Pandas wyrażenie df[...] wybiera tylko te wiersze, dla których wyrażenie w nawiasie kwadratowym zwraca True.
    Oznacza to, że wiersze, które spełniają warunek, są zachowane.

PRZYKLAD __________________________________________________________________________________________
Załóżmy, że mamy następujący DataFrame:

| A    | B    | C    |
|------|------|------|
| 1.0  | NaN  | 3.0  |
| NaN  | 2.0  | 3.0  |
| 4.0  | 5.0  | NaN  |
| NaN  | NaN  | 3.0  |

    threshold = 0.5 (50% braków):
        max_missing_allowed = 0.5 * 3 = 1.5 (maksymalnie 1 brak na wiersz).
        Liczba braków w wierszach:

Row 0: 1 brak
Row 1: 1 brak
Row 2: 1 brak
Row 3: 2 braki

Warunek logiczny:

[True, True, True, False]

Wynik:

        | A    | B    | C    |
        |------|------|------|
        | 1.0  | NaN  | 3.0  |
        | NaN  | 2.0  | 3.0  |
        | 4.0  | 5.0  | NaN  |




"""