# Module 1 Task 6
# TASK: Napisz funkcję load(fpath), która wykryje separator, wczyta plik i zwróci pd.DataFrame
import pandas as pd

def load(fpath: str) -> pd.DataFrame:  # type annotation; funkcja load zwraca obiekt klasy pd.DataFrame
    # otworzenie pliku jako plik tekstowy
    with open(fpath, 'r') as file:
        first_line = file.readline()
        second_line = file.readline()
        # print(first_line)
        # print(second_line)

    # okreslenie separatora
    if ',' in second_line:
        sep = ','  # przecinek (CSV)
    elif '\t' in second_line:
        sep = '\t'  # tabulator (TSV)
    elif ';' in second_line:
        sep = ';'  # jako separator pól bywa także stosowany znak średnika (zalezy od ustawien regionalnych systemu)
        # przy imporcie z Excel ';' -> czasami kwantyfikator rozdzielający w CSV
    else:
        raise ValueError("Nie znaleziono separatora")
    # jeśli żaden z powyzszych nie zostanie wykryty, podnoszony jest wyjątek.

    # Wczytanie pliku jako DataFrame
    return pd.read_csv(fpath, sep=sep)


# Test
if __name__ == "__main__":
    path = "./dane.csv"  # Ścieżka do pliku
    df = load(path)
    print(df)
