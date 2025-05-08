# Module 2 Task 1
"""
Baza została podzielona na wiele plików JSON nazwanych beer_1.json, beer_2.json, itd.
Rozejrzyj się w strukturze przykładowego pliku beer_1.json i
    napisz funkcję: load_beer(dirpath), która wczyta wszystkie pliki z fragmentami tej bazy danych z podanego folderu
    (argument dirpath to ścieżka do odpowiedniego folderu) i zwróci DataFrame zawierający wszystkie dane z pominięciem
    metadanych samego pliku.
-- Podpowiedź
Pliki JSON często są zagnieżdżone i zawierają więcej niż jeden obiekt. Upewnij się, że został wybrany właściwy.
-- Podpowiedź
Przypomnij sobie różnice w działaniu metod .append() i .extend().
"""
import json  # do wczytywania danych z plików JSON.
import os  # do pracy z systemem plików -> iteracja po plikach w folderze
import pandas as pd


def load_beer(dirpath: str) -> pd.DataFrame:  # dirpath: ścieżka do folderu, w którym znajdują się pliki JSON
    """
    Zadanie funkcji:
    Wczytuje dane z plików JSON w folderze, łączy je w jedną tabelę (DataFrame) i zwraca do dalszej analizy.
    Dane, które nas interesują, są zapisane w kluczu "data" każdego pliku, a inne informacje (np. metadane) są pomijane.

    PATRZ PLIK: load_beer_eplanation_t1.py
    """

    # Lista do przechowywania wczytanych danych
    data = []

    # iterowanie po wszystkich plikach w folderze
    for filename in os.listdir(dirpath):
        """
        Używamy os.listdir(dirpath) do pobrania listy wszystkich plików w folderze dirpath.
        Pętla przechodzi po nazwach tych plików jeden po drugim.
        """
        # sprawdzamy czy plik zaczyna się od "beer_" i kończy na ".json"
        if filename.startswith("beer_") and filename.endswith(".json"):
            file_path = os.path.join(dirpath, filename)  # <- SEE FILE
            print(file_path)

            # otwieranie i wczytywanie zawartości pliku JSON
            with open(file_path, 'r', encoding='utf-8') as f:
                json_content = json.load(f)

                # Jeśli JSON zawiera klucz "data", to go wybieramy (pomijając metadane)
                if 'data' in json_content:
                    data.extend(json_content['data'])  # Łączenie danych
                    """
                    Sprawdzamy, czy w załadowanym obiekcie znajduje się klucz data.
                    Jeśli tak, używamy data.extend(json_content['data']), aby dodać te dane do naszej listy data.
                    Metoda extend łączy dwie listy, zachowując strukturę listy, zamiast tworzyć zagnieżdżone listy.
                    """

    # Tworzenie DataFrame z wczytanych danych
    return pd.DataFrame(data)

dir_path = "."
common_df = load_beer(dir_path)

# Stworzenie nowego pliku JSON (JavaScript Object Notation)
common_df.to_json("t1_common_beers.json", orient='records', lines=True)