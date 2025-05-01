"""
Ten kod ilustruje proces standaryzacji danych za pomocą StandardScaler.
Standaryzacja polega na przekształceniu danych tak, aby każda kolumna miała średnią równą 0
i odchylenie standardowe równe 1. Dzięki temu cechy są w tej samej skali, co jest szczególnie
przydatne w wielu algorytmach uczenia maszynowego.

Przykład działa na danych zawierających wzrost i wagę w różnych jednostkach. Przed standaryzacją
cechy są trudne do porównania. Po standaryzacji obie kolumny są w tej samej skali.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Tworzenie przykładowego DataFrame
data = {
    "Wzrost (cm)": [150, 160, 170],
    "Waga (kg)": [50, 60, 80]
}
df = pd.DataFrame(data)

# Wyświetlenie danych przed standaryzacją
print("Dane przed standaryzacją:")
print(df)

# Tworzenie obiektu StandardScaler i standaryzacja danych
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Przekształcenie zstandaryzowanych danych z powrotem do DataFrame z oryginalnymi nazwami kolumn
scaled_df = pd.DataFrame(df_scaled, columns=df.columns)

# Wyświetlenie danych po standaryzacji
print("\nDane po standaryzacji:")
print(scaled_df)
