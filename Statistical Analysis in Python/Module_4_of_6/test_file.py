import pandas as pd

# Data reading
df = pd.read_csv("pizza.csv").drop(["brand", "id"], axis=1)

corr_matrix = df.corr()
print(corr_matrix)
'''
Otrzymany wynik macierzy korelacji, nie odnosi się do konkretnej próbki, ani nie jest średnią z próbek.
Jest to tabela współczynników korelacji obliczonych dla wszystkich par kolumn w twoim zbiorze danych.

--> Procedura obliczania macierzy korelacji: <--
Krok 1: Obliczenie korelacji dla pary (mois, prot)
        Obliczany jest współczynnik korelacji, np. korelacja Pearsona.
    średnie -> odchylenie std -> kowariancja -> współczynnik korelacji

Krok 2: Powtarzanie dla każdej pary kolumn
        Funkcja df.corr() wykonuje te same obliczenia dla wszystkich innych par kolumn:
        (mois, fat)
        (mois, ash)
        (mois, sodium)
        (prot, fat)
... i tak dalej, dla wszystkich kombinacji.

Ponieważ macierz korelacji jest symetryczna r(X,Y)=r(Y,X)),
    obliczenia dla (fat,mois) są identyczne jak dla (mois,fat).

Krok 3: Korelacja kolumny z samą sobą
    Dla każdej kolumny z samą sobą (np. (mois, mois)), korelacja wynosi 1.
    To dlatego diagonala macierzy korelacji zawsze zawiera jedynki.
    '''

most_corr = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()
print('----------------')
print(most_corr)
'''
*** Rozwinięcie macierzy korelacji do jednego poziomu:
        most_corr = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()
___unstack(): Przekształca macierz korelacji z formatu tablicowego do serii o jednej płaszczyźnie.
Przykład: Wartości z macierzy 2D stają się parą indeksów (kolumna1, kolumna2) i odpowiadającą im wartością korelacji.
___sort_values(ascending=False):
Sortuje wszystkie pary zmiennych według wartości korelacji w kolejności malejącej (największa na górze).
___drop_duplicates():
Usuwa zduplikowane pary zmiennych. Macierz korelacji jest symetryczna
    (np. corr(A,B)=corr(B,A)), więc wystarczy jedna wartość.'''

print("-----")
print(most_corr.head())
print("-----")
print(most_corr.tail())
'''
Wypisanie najbardziej skorelowanych i nieskorelowanych par zmiennych
* print(most_corr.head())
head(): Wypisuje pierwsze kilka par zmiennych, które mają najwyższą korelację.
Te zmienne są najsilniej powiązane w sposób liniowy.
* print(most_corr.tail())
tail(): Wypisuje ostatnie kilka par zmiennych, które mają najniższą korelację.
* Wartości bliskie 0 oznaczają brak korelacji.'''