def load_beer(dirpath: str) -> pd.DataFrame:
    # Lista do przechowywania wczytanych danych
    data = []

    # iterowanie po wszystkich plikach w folderze
    for filename in os.listdir(dirpath):
        # sprawdzamy czy plik zaczyna się od "beer_" i kończy na ".json"
        if filename.startswith("beer_") and filename.endswith(".json"):
            file_path = os.path.join(dirpath, filename)

            # otwieranie i wczytywanie zawartości pliku JSON
            with open(file_path, 'r', encoding='utf-8') as f:
                json_content = json.load(f)

                # Jeśli JSON zawiera klucz "data", to go wybieramy (pomijając metadane)
                if 'data' in json_content:
                    data.extend(json_content['data'])  # Łączenie danych

"""
------> Co robi ta funkcja?
Funkcja load_beer:
    * Zadanie główne:
    Wczytuje dane z plików JSON w folderze, łączy je w jedną tabelę (DataFrame) i zwraca do dalszej analizy.
    Dane, które nas interesują, są zapisane w kluczu "data" każdego pliku, a inne informacje (np. metadane) są pomijane.

    * Działania krok po kroku:
        1. Przeszukuje folder: Przechodzi przez wszystkie pliki w podanym folderze (dirpath) i wybiera te, których nazwa
         zaczyna się od "beer_" i kończy na ".json".
        2. Wczytuje zawartość plików: Otwiera każdy plik JSON i wczytuje dane do obiektu Python (np. słownika).
        3. Filtruje dane: Z każdego pliku wybiera tylko te dane, które znajdują się w kluczu "data"
        (pomijając np. klucz "metadata").
        4. Łączy dane: Wszystkie rekordy z plików JSON są dodawane do jednej listy.
        5. Tworzy tabelę: Na koniec lista zebranych danych jest przekształcana w tabelę (df) z kolumnami i wierszami.

    * Co dostajemy na wyjściu?
    Funkcja zwraca jeden obiekt DataFrame, który zawiera połączone dane ze wszystkich plików JSON w folderze.
    DOSTAJEMY JEDEN DUZY DATAFRAME



------> Struktura pliku JSON
    {
        "currentPage": 1,
        "numberOfPages": 603,
        "totalResults": 30121,
        "data": [
            {"id": 1, "name": "Golden Ale", "rating": 4.5},
            {"id": 2, "name": "Pale Ale", "rating": 4.0}
        ],
        "status": "success"
    }
Funkcja load_beer:
    Ignoruje:           Klucze takie jak "currentPage", "numberOfPages", "totalResults", i "status".
    Przetwarza tylko:   Zawartość klucza "data", który jest listą rekordów;
    
    
    
------>  Czy os.path.join automatycznie dodaje "/" ?
Tak, os.path.join automatycznie dodaje separator ścieżki (/ lub \, w zależności od systemu operacyjnego),
jeśli jest to konieczne. 

    Linux/Mac (separator /):
        import os
        dirpath = "/home/username"
        filename = "file.txt"
        path = os.path.join(dirpath, filename)
        print(path)
    Wynik:   /home/username/file.txt

Czy doda separator w każdym przypadku?
    Jeśli pierwszy fragment kończy się separatorem (/ lub \), os.path.join nie doda dodatkowego separatora.



------>  Jakie sa roznice w dzialaniu metod .append() i .extend().?
Metody .append() i .extend() w Pythonie są używane do manipulacji listami, ale mają różne zastosowania.

1. .append()____________________
Dodaje pojedynczy element na koniec listy jako całość.
Jak działa?
    - Traktuje podany argument jako jeden obiekt i dodaje go do listy.
    - Jeśli argumentem jest lista, zostanie dodana jako element zagnieżdżony (lista wewnątrz listy).

Przykład:
    my_list = [1, 2, 3]
    my_list.append(4)
    print(my_list)
# Wynik: [1, 2, 3, 4]
    
# Dodanie listy jako jednego elementu
    my_list.append([5, 6])
    print(my_list)
# Wynik: [1, 2, 3, 4, [5, 6]]  <---- PATRZ

2. .extend()____________________
Rozszerza listę, dodając elementy z innej sekwencji (np. innej listy) jako osobne elementy.
Jak działa?
    - Traktuje argument jako iterowalny obiekt (np. listę, krotkę, ciąg znaków).
    - Dodaje każdy element tego iterowalnego obiektu osobno do listy.

Przykłady:
    my_list = [1, 2, 3]
    my_list.extend([4, 5, 6])
    print(my_list)
# Wynik: [1, 2, 3, 4, 5, 6]

# Dodanie ciągu znaków
    my_list.extend("AB")
    print(my_list)
# Wynik: [1, 2, 3, 4, 5, 6, 'A', 'B']  <---- PATRZ
"""
