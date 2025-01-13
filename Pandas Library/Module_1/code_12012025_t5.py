# Module 1 Task 5
# zwróc liczbę naturalną odpowiadającą liczbie wierszy w pierwszych i ostatnich nrows wierszach, w których przynajmniej
# jedna wartość jest mniejsza od thresh.
import pandas as pd

def less_than(df, nrows, thresh):
    # pobieranie- pierwsze nrows wierszy
    head_rows = df.head(nrows)
    # pobieranie- ostatnie nrows wierszy
    tail_rows = df.tail(nrows)
    # Łączymy oba fragmenty DataFrame w jeden
    total_rows = pd.concat([head_rows, tail_rows])
    # sprawdzenie wartosci mniejszych niż thresh
    condition = (total_rows < thresh).any(axis=1) # see note
    # suma wierszy, które spełniają warunek
    result = condition.sum()
    return result

if __name__ == "__main__":
    test_df = pd.read_csv("dane.csv")
    n_rows = 20
    threshold = 0
    less_than(test_df, n_rows, threshold)
    print(less_than(test_df, n_rows, threshold))

'''
-> Metoda .any(axis=1) w Pandas sprawdza, czy w każdym wierszu DataFrame lub Series występuje co najmniej jedna wartość
spełniająca dany warunek.
-> Argument axis=1: oznacza, że operacja odbywa się po wierszach (poziomo).
                    gdy axis=0, sprawdzenie odbywa się po kolumnach (pionowo).

Z    .any(axis=1):   Wynik to seria logiczna dla wierszy, a .sum() daje pojedynczą liczbę.
Bez  .any(axis=1):   Wynik to pełny DataFrame logiczny, a .sum() zwraca sumy dla każdej kolumny osobno.
'''

def less_than2(df, nrows, thresh):
    head_rows = df.head(nrows)
    tail_rows = df.tail(nrows)
    total_rows = pd.concat([head_rows, tail_rows])
    condition = (total_rows < thresh)  #.any(axis=1) #
    result = condition.sum()
    return result

if __name__ == "__main__":
    test_df = pd.read_csv("dane.csv")
    n_rows = 20
    threshold = 0
    print('')
    print(less_than2(test_df, n_rows, threshold))