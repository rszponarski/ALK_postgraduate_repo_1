# Module 1 Task 9
"""Napisz funkcję fill_missing_categorical_with_mode(df), która wykryje kolumny z danymi kategorialnymi
i zastąpi występujące w nich brakujące wartości najczęściej występującą wartością w danej kolumnie. """
import pandas as pd


def fill_missing_categorical_with_mode(df: pd.DataFrame) -> pd.DataFrame:
    """
    -> Funkcja przyjmuje na wejsciu dataframe;
    -> wykryje kolumny z danymi kategorialnymi i zastąpi występujące w nich
    brakujące wartości najczęściej występującą wartością w danej kolumnie.
    """
    # stworzenie kopii DataFrame, aby nie modyfikować oryginału
    new_df = df.copy()

    # iteracja przez wszystkie kolumny df
    for col in new_df.columns:
        # sprawdzenie czy kolumna jest typu obiektowego
        if new_df[col].dtype == 'object':
            # obliczenie mody(dominanty) kazdej kolumny = najczęściej występującą wartość
            mode_col = new_df[col].mode().iloc[0] # <- SEE NOTE
            # print(f"MODE: {mode_col}")
            # zastapienie NaN wartością najczęściej występującą
            # new_df[col].fillna(mode_col, inplace=True) #<-- SEE FUTURE WARNING BELOW
            new_df[col] = new_df[col].fillna(mode_col) # <-- USE eg. THIS

    return new_df


# test
if __name__ == "__main__":
    # read csv file
    test_df = pd.read_csv("./dane1401.csv")
    # check where is NaN
    empty_mask = test_df.isna()
    empty_cells = empty_mask.stack()[empty_mask.stack()] # wybiera tylko te elementy, które są True (czyli z NaN).
    print(f"EMPTY CELLS:\n{empty_cells}\n")

    print(test_df.loc[27, "col6"]) # <- uzywajac .loc wpisuj dokladnie ten indeks row, ktory sie wyswietla (nie -1)
    """ EMPTY CELLS:
        27  col6     True"""

    # fill NaN with NaN
    filled_df = fill_missing_categorical_with_mode(test_df)
    print(filled_df.loc[27, "col6"])

'''NOTES:
ad. mode() _______________________________________

df[col].mode():
- Funkcja .mode() zwraca serię z wartością/ami, które występują najczęściej w kolumnie.
- Jeśli jest więcej niż jedna moda (np. wartości A i B występują po równo najczęściej), .mode() zwraca
  wszystkie te wartości jako serię.
Przykład: Jeśli kolumna ['A', 'A', 'B', 'NaN', 'B', 'C'], to .mode() zwróci ['A', 'B'].
- .iloc[0]: Pobiera pierwszą wartość z serii zwróconej przez .mode(). Rozwiązanie radzi sobie z przypadkiem wielomody
   (gdy jest więcej niż jedna najczęstsza wartość). ----> .iloc[0] zwróci 'A'.

ad. FUTURE WARNING _______________________________

Ten FutureWarning w Pandas informuje, że metoda inplace=True używana na wycinku DataFrame (np. df[col])
może w przyszłości nie działać. 

Zastąp linijkę ---> new_df[col].fillna(mode_col, inplace=True) :
    -> new_df[col] = new_df[col].fillna(mode_col)
LUB
    -> new_df.fillna({col: mode_col}, inplace=True)
'''