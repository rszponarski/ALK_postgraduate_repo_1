# Module 2 Task 1
"""
Napisz funkcję load_random_subset(filepath, size),
która wczyta dane w formacie JSON Lines i zwróci DataFrame z losowym podzbiorem size wierszy.
"""
import pandas as pd

def load_random_subset(filepath: str, size: int) -> pd.DataFrame:
    # Wczytanie danych z pliku JSON Lines
    data = pd.read_json(filepath, lines=True)
    """
    Metoda pd.read_json() z opcją lines=True wczytuje plik w formacie JSON Lines.
    Każda linia jest traktowana jako osobny obiekt JSON.
    """
    # Losowe próbkowanie wierszy, z uwzględnieniem wielkości zbioru
    return data.sample(n=size, replace=True if size > len(data) else False)
"""
    __n=size określa, ile wierszy chcemy wylosować.
    __replace pozwala na losowanie z powtórzeniami:
    
    __replace=True: Jeśli żądana liczba wierszy (size) jest większa niż liczba dostępnych wierszy w zbiorze danych,
    to losowanie odbywa się z powtórzeniami.
    __replace=False: Jeśli żądana liczba wierszy jest mniejsza lub równa, to losowanie odbywa się bez powtórzeń.
"""

file_path = "beer.jsonl"
my_random_subset = load_random_subset(file_path, 5)
print(my_random_subset)
print(type(my_random_subset))


"""
NOTES:

 JSON:
 - jest prosty do zrozumienia,
 - czytelny dla osób bez doświadczenia w programowaniu i bardzo dobrze ustrukturyzowany.
 
JSON MA kluczową wadę: ---> nie najlepiej nadaje się do przetwarzania danych po kawałku.
W odpowiedzi na te problemy powstał format JSON Lines (.jsonl)
- Zapisuje on dane w formacie „jedna linia — jeden obiekt JSON", co pozwala na odczyt fragmentami.
Dzięki temu można analizować duże zbiory danych, odczytując je stopniowo, bez konieczności ładowania całości do pamięci.
Ułatwia to też pracę z danymi w czasie rzeczywistym (np. z logami serwera, klastra czy aplikacji)
oraz planowanie obliczeń równoległych na wielu maszynach naraz.
---------------------------------------------------------------------------------------------------

Główne parametry data.sample()
1. n

    Liczba losowych wierszy (lub kolumn), które chcesz wybrać.
    Jeśli wartość n jest większa niż liczba dostępnych wierszy (lub kolumn), musisz ustawić parametr replace=True,
    aby losować z powtórzeniami.

2. frac

    Frakcja (ułamek) całkowitej liczby wierszy (lub kolumn), którą chcesz losowo wybrać.
    Na przykład, jeśli frac=0.5, to wybierzesz 50% danych.
    Parametry n i frac są wzajemnie wykluczające — możesz użyć tylko jednego z nich.

3. replace
    Domyślnie: False.
    Jeśli ustawione na True, losowanie odbywa się z powtórzeniami, co oznacza, że ten sam wiersz (lub kolumna)
    może pojawić się więcej niż raz w wyniku.

4. random_state

    Ustalanie ziarna dla generatora liczb losowych, aby wyniki były powtarzalne.
    Przydatne w sytuacjach, gdy potrzebujesz odtworzyć wyniki (np. w badaniach).

5. axis

    Domyślnie: 0 (losuje wiersze).
    Jeśli ustawisz axis=1, losuje kolumny zamiast wierszy.
"""