# Module 2 Task 6
import pandas as pd

def clean_dataframe(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    FUNKCJA: pobiera df (do oczyszczenia ) i threshold.
    Podstawowy preprocessing, ktory sie wykonuje w tej funkcji -> Funkcja:
    -> Usunie z DataFrame'u wiersze, w których brakuje więcej niż threshold * 100% danych,
    -> Usunie wszystkie duplikaty (wiersze, które są identyczne jak wcześniejszy wiersz),
    -> Wykryje kolumny z wartościami boolowskimi zapisanymi jako string i zamieni je na typ bool.

    --- threshold (float): Próg brakujących danych dla usunięcia wierszy.
    """
    # jaka jest minimalna liczba niebrakujacych wierszy?
    min_non_missing = int((1 - threshold) * df.shape[1])
    # usuniecie tych wierszy, gdzie liczba brakujących wartości wieksza niż threshold
    df = df.dropna(thresh=min_non_missing)
        # zachowaj tylko wiersze zawierające co najmniej min_non_missing wartości innych niż NaN
    # usuniecie zduplikowanych wierszy
    df = df.drop_duplicates()
    # wykrywanie kolumn zawierających wartości boolowskie zapisane jako string
    for col in df.columns:
        if df[col].dtype == 'object':
            # sprawdzam, czy w kolumnie występuje przynajmniej jedno "true" lub "false"-> zakladam, ze
            # to powinna byc kolumna z dtypes: boolean
            if df[col].str.lower().isin(["true", "false"]).any():
                # zmiana typu kolumny na bool
                df[col] = df[col].apply(lambda x: str(x).lower() == "true")
    return df

if __name__ == "__main__":
    # wczytywanie danych z pliku
    beer_df = pd.read_json("flat_beer.jsonl", lines=True)
    # czyszczenie danych za pomoca funcji clean_dataframe(); treshold: 0.5
    cleaned_df = clean_dataframe(beer_df, threshold=0.5)
    # wyswietlenie danych
    print(cleaned_df.head(3))
    print(cleaned_df.tail(3))
    print(cleaned_df["true/false"].dtypes)

"""
min_non_missing = int((1 - threshold) * df.shape[1])
df.shape[1]: zwraca liczbę kolumn w DataFrame -> df.shape zwraca tuple (0,1)=(wiersze, kolumny) 
max_missing), które są dopuszczalne w jednym wierszu. -> np. 0.5 * 10 = 5 -> usuwamy wiersz jesli 5 kolumn 
                                                                             nie jest uzupelnionych
df.isnull().sum(axis=1) oblicza liczbę brakujących wartości w każdym wierszu.
df[...] filtruje DataFrame i pozostawia tylko wiersze spełniające warunek.
"""

"""
if df[column].dropna().apply(lambda x: str(x).lower()).isin(["true", "false"]).all():
    df[column] = df[column].apply(lambda x: str(x).lower() == "true")

Trzeba sprawdzic, czy kolumna zawiera wartości logiczne zapisane jako string
--df[column].dropna(): 1-- usuwamy wartości NaN z kolumny <by nie powodowały błędów dalej>;
--.apply(lambda x: str(x).lower()): 2-- Kazda wartosc przeksztalcamy na string i zmieniamy
na małe litery (np. "True" → "true").
--.isin(["true", "false"]): 3-- sprawdzenie, czy każda wartość w kolumnie należy do zbioru
["true", "false"].
--.all(): zapewnia, że warunek (wartości w kolumnie to tylko "true" lub "false") DOTYCZY
KAZDEJ WARTOSCI w kolumnie. Jeśli go nie dodamy, dostaniemy serię wartości logicznych
zamiast jednego wyniku True lub False. -> chodzi o to zeby byc pewnym, ze mozemy
przekonwertowac cala te kolumne na bool.
"""
"""
Konwertowanie wartości na typ bool
Jeśli warunek w if jest spełniony, każda wartość w kolumnie jest konwertowana na bool:
lambda x: str(x).lower() == "true": Zamienia string "true" na True i "false" na False.

---> Dlaczego wartości stają się bool?
Działanie porównania (==)
Wyrażenie str(x).lower() == "true" zwraca wynik typu logicznego (bool):
Jeśli porównanie jest prawdziwe, wynik to True.
Jeśli jest fałszywe, wynik to False.

Metoda .apply() stosuje funkcję do każdej wartości w kolumnie.
"""