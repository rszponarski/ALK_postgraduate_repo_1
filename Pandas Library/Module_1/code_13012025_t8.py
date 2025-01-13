# Module 1 Task 8
"""Napisz funkcję fill_missing_with_mean(df), która wykryje kolumny z danymi numerycznymi, znajdzie w nich brakujące
wartości i zastąpi je średnią z pozostałych wartości w danej kolumnie.
--Funkcja powinna zwrócić kopię wczytanego DataFrame z oczyszczonymi danymi.
--Przyjmij, że DataFrame może zawierać tylko liczby 64-bitowe."""
import pandas as pd


def fill_missing_with_mean(df: pd.DataFrame) -> pd.DataFrame:
    """
    FUNKCJA: Zastępuje brakujące wartości w kolumnach numerycznych średnią z pozostałych wartości.
    Funkcja pobiera istniejacy DataFrame.
    Otrzymujemy: pd.DataFrame: Nowy DataFrame z uzupełnionymi wartościami.
    """
    # utworzenie kopii oryginalnego DataFrame, aby nie modyfikować go bezpośrednio
    new_df = df.copy()

    # iterowanie przez kolumny o typie float64 i int64
    for col in new_df.select_dtypes(include=['float64', 'int64']).columns:
        # obliczenie średniej dla danej kolumny, ignorując NaN
        mean_value = new_df[col].mean()
        # zastapienie brakujących wartości średnią
        new_df[col].fillna(mean_value, inplace=True)

    return new_df


if __name__ == "__main__":
    my_df = pd.read_csv("./raw_data.csv")

    print(my_df.columns) # df.columns --> wyswietli tylko nazwy kolumn
    print(my_df.dtypes)  # Wyświetlanie nazw kolumn i ich typów danych
    print('')
    # po sprawdzeniu pliku csv wiem, ze np. okno [150 x "col99"] jest NaN; sprawdzmy:
    print(my_df.loc[150, "col99"])  # zgadza sie
    # teraz zastosujmy nasza funkcje:
    brand_new_df = fill_missing_with_mean(my_df)
    print(brand_new_df.loc[150, "col99"]) # <-- teraz zwraca wartosc srednia pozostalych wartosci w kolumnie
    # wiec analogicznie nastepny wiersz, ktory tez byl NaN:
    print(brand_new_df.loc[151, "col99"])

"""NOTES:
---> Co oznacza dtypes?
Atrybut dtypes: Zwraca typ danych każdej kolumny w DataFrame jako obiekt Series.
Typy danych (np. int64, float64, object) wskazują na sposób przechowywania danych w każdej kolumnie:
    -- int64: Liczby całkowite (64-bitowe).
    -- float64: Liczby zmiennoprzecinkowe (64-bitowe).
    -- object: Dane tekstowe (lub mieszane, np. tekst i liczby w jednej kolumnie).
    -- Inne typy mogą obejmować bool, datetime64, czy category.

DLATEGO w funkcji fill_missing_with_mean iterujemy przez wszystkie kolumny, sprawdzajac typ danych
    Filtrowanie kolumn według typu danych (używane w kombinacji z select_dtypes()):
            
            df.select_dtypes(include=['int64', 'float64'])
"""