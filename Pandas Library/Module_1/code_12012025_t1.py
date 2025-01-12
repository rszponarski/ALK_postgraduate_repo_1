# Module 1 Task 1
import numpy as np
import pandas as pd

# przykładowe dane
n_rows = 5
l_columns = [1, 2, 3, 4, 5]
nan = False


def dummy_data(n, columns, do_nan=False):
    if do_nan:  # do_nan==True
        data = np.full((n, len(columns)), np.nan)
        # tworzy tablicę o wymiarach (n, len(columns)) wypełnioną wartością np.nan.
    else:
        data = np.random.rand(n, len(columns))
        # generuje dwuwymiarową tablicę numpy o rozmiarach n x len(columns);
        # funkcja np.random.rand (z NumPy) generuje losowe liczby zmiennoprzecinkowe w przedziale:
        # Od 0.0 (włącznie)--> Do 1.0 (wyłącznie).
    return pd.DataFrame(data, columns=columns)


# Test_1
df = dummy_data(n_rows, l_columns, nan)
print(df)

# Test_2
nan = True
df = dummy_data(n_rows, l_columns, nan)
print(f"\n{df}")
