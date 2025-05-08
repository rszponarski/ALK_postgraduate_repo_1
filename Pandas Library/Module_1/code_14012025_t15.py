# Module 1 Task 14
"""
 Napisz funkcję, która wyeksportuje Twój df do plików tymczasowych w trzech różnych formatach: CSV, TSV, i JSON.
 Następnie niech porówna ich wielkości i zwróci nazwę formatu, który zajmuje najmniej miejsca.

Poza biblioteką Pandas może przydać Ci się biblioteka os.
Ciekawostka: Jeśli nazwa pliku zaczyna się od ., to będzie on ukryty w systemie operacyjnym, dopóki wprost się do niego
nie odwołasz. To konwencja często używana w przypadku plików konfiguracyjnych i tymczasowych. """

import os  # pozwala na dostęp do operacji systemowych, takich jak sprawdzanie rozmiaru plików.
import pandas as pd

def export_and_compare_sizes(df: pd.DataFrame, base_filename: str = ".exported_file") -> str:
    # domyślnie przyjmuje wartość "./exported_file" jako nazwę bazową pliku eksportowanego.
    # domyślnie, jeśli nie podasz innej nazwy pliku, zapis pliku będzie ukryty,

    # CSV Export
    csv_filename = f"{base_filename}.csv"
    df.to_csv(csv_filename, index=False)
    """
    - Tworzymy nazwę pliku CSV poprzez dodanie rozszerzenia .csv do base_filename.
    - df.to_csv() eksportuje df do pliku CSV. Parametr index=False sprawia, że indeks nie zostaje zapisany w pliku.
    """

    # TSV Export
    tsv_filename = f"{base_filename}.tsv"
    df.to_csv(tsv_filename, index=False, sep='\t')
    """
    - Parametr sep='\t' używa tabulacji jako separatora kolumn.
    """

    # JSON Export
    json_filename = f"{base_filename}.json"
    df.to_json(json_filename, orient='records', lines=True)
    """
    - df.to_json() eksportuje DataFrame do pliku JSON.
    - Parametr orient='records' oznacza, że każdy rekord jest pojedynczym elementem wiersza;
    - lines=True zapisuje każdy rekord w osobnym wierszu.
    """

    # Porównanie rozmiarów plików
    file_sizes = {
        csv_filename: os.path.getsize(csv_filename),
        tsv_filename: os.path.getsize(tsv_filename),
        json_filename: os.path.getsize(json_filename),
    }
    print("\n", file_sizes)
    """
    - Funkcja os.path.getsize() oblicza rozmiar każdego z plików w bajtach.
    - Tworzymy słownik file_sizes z nazwami plików jako kluczami oraz ich rozmiarami jako wartościami.
    """

    # Znalezienie najmniejszego pliku
    min_size_file = min(file_sizes, key=file_sizes.get)
    """
    Funkcja min() znajduje klucz (nazwa pliku) o najmniejszym rozmiarze.
    Funkcja zwraca nazwę pliku o najmniejszym rozmiarze.
    """

    return min_size_file

if __name__ == "__main__":
    books_df = pd.read_csv("./books_t15.csv")
    best_format = export_and_compare_sizes(books_df, "TASK15_exported")
    print("\n", best_format)

"""
PODSUMOWANIE:
Funkcja stworzyła trzy pliki (CSV, TSV, JSON), ponieważ każda operacja eksportu jest niezależna i odbywa się na
podstawie różnych metod eksportu.
Celem zadania bylo, że funkcja zwróci tylko nazwę pliku o najmniejszym rozmiarze.
"""