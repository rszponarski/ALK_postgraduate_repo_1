# Module 2 Task 7
"""
Gdy brakuje wielu pomiarów, powielając średnią (i uzupelniajac nia braki w danych), za bardzo oddalamy się od
prawdziwego rozkładu próby. Są jednak inne metody, które pozwalają na zachowanie większej ilości informacji.
Jedną z nich jest użycie algorytmów do imputacji danych. W tym zadaniu trzeba użyć jednego z nich.

GLOWNA ROZNICA MIEDZY IMPUTACJA A INTERPOLACJA polega na założeniu, że kolejność ma znaczenie (dla interpolacji).
W przypadku imputacji nie zakładamy, że dane są uporządkowane. W związku z tym, imputacja jest bardziej uniwersalna,
ale też bardziej skomplikowana, bo nie wystarczy znajomość wartości sąsiednich punktów.

Napisz funkcję fill_missing_values(df, target_column, feature_column), która wykorzystuje regresję liniową,
by imputować brakujące wartości w jednej kolumnie na podstawie innej, wybranej kolumny.
Skorzystaj z gotowej implementacji regresji liniowej LinearRegression z modułu biblioteki scikit-learn.linear_model.
Przetestuj swoją funkcję, imputując brakujące wartości z kolumny IBU (międzynarodowa jednostka goryczy) na podstawie
wartości z kolumny ABV (zawartość alkoholu).

Aby poprawnie imputować dane jednej kolumny na podstawie drugiej, musisz na początku dopasować (ang. fit)
gotową implementację regresji liniowej do swoich danych.
"""
import pandas as pd
from sklearn.linear_model import LinearRegression # LinearRegression z sklearn.linear_model: Klasa implementująca
# liniowy model regresji, który znajdzie liniową zależność między cechą (feature_column) a wartością docelową
# (target_column).

def fill_missing_values(df: pd.DataFrame, target_column: str, feature_column: str) -> pd.DataFrame:
    """
    target_column:  Nazwa kolumny z brakującymi wartościami, które chcemy imputować (uzupełniać).
    feature_column: Nazwa kolumny używanej jako cecha (zmienna niezależna) do przewidywania brakujących wartości.

    Funkcja zwraca DataFrame z uzupełnionymi wartościami w target_column.
    """

    # Oddziel dane z brakującymi wartościami i bez nich
    df_complete = df.dropna(subset=[target_column, feature_column])
    df_missing = df[df[target_column].isnull()]
    """
    df_complete:  Tworzymy nowy DataFrame zawierający tylko wiersze, w których zarówno target_column,
                  jak i feature_column nie mają braków (NaN).
    df_missing:   Tworzymy DataFrame zawierający tylko te wiersze, w których target_column ma braki (NaN).
                  Te wartości zostaną uzupełnione. """

    # Podział danych na cechę i target <-- PRZYGOTOWANIE DANYCH DO TRENOWANIA MODELU
    X_train = df_complete[[feature_column]]
    y_train = df_complete[target_column]
    """
    X_train:    Kolumna cechy (feature_column) z df_complete, która będzie używana jako wejście (zmienna niezależna) do
    modelu. --> Zauważ, że podwójne nawiasy [[...]] sprawiają, że X_train pozostaje DataFrame'em (a nie Series).

    y_train:    Kolumna docelowa (target_column) z df_complete, która będzie używana jako wyjście (zmienna zależna) do
    modelu.
    """

    # Dopasowanie modelu <-- TRENOWANIE MODELU
    model = LinearRegression()  # Tworzymy instancję modelu regresji liniowej. <- patrz NOTES
    model.fit(X_train, y_train)
    """
    Model uczy się związku między feature_column (wejściem) a target_column (wyjściem).
    Szuka prostej linii (modelu liniowego), która najlepiej dopasowuje się do danych.
    """

    # Imputacja brakujących wartości <-- Przewidywanie brakujących wartości
    if not df_missing.empty:
        X_missing = df_missing[[feature_column]]
        df.loc[df[target_column].isnull(), target_column] = model.predict(X_missing)
    """
    -> Sprawdzamy, czy istnieją brakujące wiersze: Jeśli df_missing nie jest pusty, to kontynuujemy proces imputacji.
    
    -> X_missing = df_missing[[feature_column]]: Pobieramy kolumnę cechy (feature_column) z wierszy z brakującymi
                                                 wartościami (df_missing).
                                                    
    -> model.predict(X_missing): Model używa dopasowanej linii (z danych treningowych) do przewidzenia wartości
                                 brakujących w target_column na podstawie wartości w X_missing.

    -> Uzupełnianie brakujących wartości:  df.loc[df[target_column].isnull(), target_column]:
            Wiersze z brakami w target_column są wypełniane przewidywanymi wartościami z model.predict(X_missing).
    """
    return df  # Zwracanie uzupełnionego DataFrame


if __name__ == "__main__":
    target = "ibu"  # uzupelnij dane w kolumnie IBU na podstawie kolumny ABV
    feature = "abv"
    beer_df = pd.read_json("flat_beer_2.jsonl", lines=True)
    print(beer_df["ibu"].head(10))

    # Tworzenie kopii DataFrame <-- tego wczesniej nie bylo, wyjasnienie w uwagach na koncu
    beer_df_copy = beer_df.copy()

    # Wypełnij brakujące wartości
    filled_df = fill_missing_values(beer_df_copy, target_column=target, feature_column=feature)

    # Wyświetl pierwsze wiersze
    print(beer_df["ibu"].head(10))
    print(filled_df["ibu"].head(10))

    # print(filled_df.head())

""" NOTES

model = LinearRegression()

    LinearRegression() to konstruktor klasy LinearRegression z biblioteki scikit-learn.
    Tworzy nowy obiekt (instancję) modelu regresji liniowej, który będzie później używany do dopasowywania (fit) danych
    i przewidywania (predict).

---> Inicjalizuje model z domyślnymi parametrami

    LinearRegression() ma wbudowane parametry, które domyślnie są ustawione na typowe wartości.
    Oto niektóre z najważniejszych parametrów:
        * fit_intercept=True:   Czy model ma obliczyć wyraz wolny (przecięcie z osią Y)? Jeśli False, to linia regresji
                                będzie przechodziła przez punkt (0, 0).
        * normalize=False:      Czy dane wejściowe mają być znormalizowane przed dopasowaniem? (Domyślnie False,
                                normalizację trzeba przeprowadzić samodzielnie).
        * n_jobs=None:          Liczba wątków procesora, które mają być użyte do obliczeń.
                                (Domyślnie None, co oznacza, że model używa jednego wątku).

---> Model nie jest jeszcze trenowany

    - W momencie wywołania LinearRegression() tworzysz "pusty" model, który nie zna jeszcze żadnych danych ani nie zna 
    zależności między zmiennymi.
    - Samo LinearRegression() nie wykonuje żadnych obliczeń.
    Dopiero w momencie wywołania metody fit() model jest trenowany na danych.

Podsumowanie
    
    --> LinearRegression(): Tworzy obiekt modelu z domyślnymi (lub określonymi) parametrami.
    --> fit(): Dopasowuje model do danych, ucząc się współczynników regresji.
    --> predict(): Używa dopasowanego modelu do przewidywania wartości na podstawie nowych danych.

Porównanie do codziennego życia
    
    Myśl o LinearRegression() jako o zakupie nowego kalkulatora.
    Dopóki go nie użyjesz (fit()), jest pusty i gotowy do pracy, ale nic jeszcze nie oblicza.
    Dopiero po wprowadzeniu danych (np. wpisaniu liczb na kalkulatorze),
    kalkulator wykonuje operacje i daje wynik (predict()).
"""

""" UWAGA!
Ten kod nadpisuje wartości w beer_df zamiast zapisywać zmodyfikowany DataFrame w filled_df, ponieważ wewnątrz funkcji
fill_missing_values modyfikujesz DataFrame df in-place. 
W tym fragmencie funkcji:

df.loc[df[target_column].isnull(), target_column] = model.predict(X_missing)

    *** Funkcja używa operatora przypisania = z indeksem .loc[] do zmiany wartości w oryginalnym DataFrame df.
    W Pythonie, obiekt DataFrame jest przekazywany do funkcji przez odniesienie, a nie przez kopię.
    *** Oznacza to, że jakiekolwiek zmiany dokonane na df wewnątrz funkcji będą miały bezpośredni wpływ na pierwotny
    DataFrame (tutaj: beer_df).

Rozwiązanie:
Aby tego uniknąć, można utworzyć kopię DataFrame w funkcji lub poza nią. Oto dwa możliwe rozwiązania:

    Rozwiązanie 1:
    Tworzenie kopii na początku funkcji
    Zmodyfikuj funkcję fill_missing_values, aby utworzyć kopię wejściowego DataFrame:
    
    def fill_missing_values(
        df: pd.DataFrame, target_column: str, feature_column: str
    ) -> pd.DataFrame:
        # Utwórz kopię DataFrame
        df = df.copy()
    
        # Oddziel dane z brakującymi wartościami i bez nich
        df_complete = df.dropna(subset=[target_column, feature_column])
        df_missing = df[df[target_column].isnull()]
    
        # Podział danych na cechę i target
        X_train = df_complete[[feature_column]]
        y_train = df_complete[target_column]
    
        # Dopasowanie modelu
        model = LinearRegression()
        model.fit(X_train, y_train)
    
        # Imputacja brakujących wartości
        if not df_missing.empty:
            X_missing = df_missing[[feature_column]]
            df.loc[df[target_column].isnull(), target_column] = model.predict(X_missing)
    
        return df
    
    Rozwiązanie 2:
    Tworzenie kopii poza funkcją
    Utwórz kopię DataFrame bez modyfikowania funkcji:
    
    if __name__ == "__main__":
        target = "ibu"
        feature = "abv"
        beer_df = pd.read_json("flat_beer_2.jsonl", lines=True)
    
        # Tworzenie kopii DataFrame
        beer_df_copy = beer_df.copy()
    
        print(beer_df_copy["ibu"].head(25))
    
        # Wypełnij brakujące wartości
        filled_df = fill_missing_values(beer_df_copy, target_column=target, feature_column=feature)
    
        # Wyświetl pierwsze wiersze
        print(filled_df["ibu"].head(25))
----------------------------------------
Podsumowanie

    - LinearRegression():  Tworzy obiekt modelu z domyślnymi (lub określonymi) parametrami.
    - fit():  Dopasowuje model do danych, ucząc się współczynników regresji.
    - predict():  Używa dopasowanego modelu do przewidywania wartości na podstawie nowych danych.

"""