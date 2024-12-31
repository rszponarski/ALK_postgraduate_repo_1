import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Wczytanie danych
df = pd.read_csv('pizza.csv').drop(['id', 'brand'], axis=1)

# Standaryzacja danych
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# PCA
pca = PCA(n_components=6)
pca.fit(df_scaled)

# Obliczanie wartości składowych głównych
eigenvalues = pca.explained_variance_
prop_var = eigenvalues / np.sum(eigenvalues)

# Wyświetlanie wyników
for i, (eigen, prop) in enumerate(zip(eigenvalues, prop_var), start=1):
    print(f"Składowa PC{i}:")
    # print(f"  Wartość własna (eigenvalue): {eigen}")
    print(f"  Proporcja wyjaśnianej wariancji: {prop:.4f}")

''' OTRZYMANY WYNIK:
Składowa PC1:
  Proporcja wyjaśnianej wariancji: 0.5960
Składowa PC2:
  Proporcja wyjaśnianej wariancji: 0.3272
Składowa PC3:
  Proporcja wyjaśnianej wariancji: 0.0592
Składowa PC4:
  Proporcja wyjaśnianej wariancji: 0.0136
Składowa PC5:
  Proporcja wyjaśnianej wariancji: 0.0040
Składowa PC6:
  Proporcja wyjaśnianej wariancji: 0.0000
----
Proporcja wyjaśnianej wariancji:
    PC1: 59.60%
    PC2: 32.72%
    PC3: 5.92%
    PC4: 1.36%
    PC5: 0.40%
    PC6: 0.00%
Wnioski:
    Pierwsze 3 składowe (PC1, PC2, PC3) wyjaśniają większość wariancji w danych:

    PC1 wyjaśnia 59.60% całkowitej wariancji.
    PC2 wyjaśnia kolejne 32.72%, co razem daje 92.32% całkowitej wariancji.
    PC3 dodaje 5.92%, ale po dodaniu tej składowej (zaledwie łącznie 98.24% wariancji) dalsze składowe mają już bardzo
    mały wpływ na dane.
Składowe PC4, PC5, PC6 mają bardzo mały wpływ na dane:

    PC4 wyjaśnia tylko 1.36% wariancji.
    PC5 wyjaśnia tylko 0.40%.
    PC6 nie wyjaśnia żadnej istotnej wariancji (0.00%).
Interpretacja:
    * Składowe PC1, PC2 i PC3: Te składowe mają znaczący wpływ na strukturę danych,
    ponieważ razem wyjaśniają 98.24% całkowitej wariancji. W związku z tym,
    te trzy składowe są najbardziej istotne do dalszej analizy.
    * Składowe PC4, PC5 i PC6: Ich wpływ na dane jest minimalny, więc w praktyce można je odrzucić w dalszej analizie
    bez utraty zbyt wielu informacji.
    Te składowe mogą zostać uznane za „szum” w danych, ponieważ ich wkład w wyjaśnianie zmienności jest znikomy.

Ogólny wniosek:
Można uznać, że tylko pierwsze 3 składowe (PC1, PC2, PC3) są istotne, a resztę można odrzucić,
ponieważ ich wkład w wyjaśnianie wariancji jest marginalny.
W praktyce, wybierając mniejszą liczbę składowych głównych, np. 3, można znacząco uprościć analizę,
zachowując jednocześnie większość informacji o danych.
------------------------------------------------------

Wyjaśnianie wariancji:

Składowa główna (PC) to nowa oś w przestrzeni danych, która najlepiej odwzorowuje zmienność danych wzdłuż tej osi.
Proporcja wyjaśnianej wariancji to odsetek całkowitej zmienności danych, który jest wyjaśniany przez konkretną składową.
Na przykład:
Jeśli PC1 wyjaśnia 59.60% wariancji, oznacza to, że ta składowa potrafi uchwycić 59.60% całej zmienności danych w
zbiorze. Można to interpretować tak, że PC1 jest najważniejszą składową i "najlepiej tłumaczy" rozkład danych.
Jeśli PC2 wyjaśnia 32.72% wariancji, to ta składowa uchwyciła dodatkowe 32.72% zmienności,
ale wciąż jest to mniej niż PC1.

Wyjaśnianie wariancji w praktyce:

Załóżmy, że mamy dane o różnych cechach, takich jak waga, wzrost, wiek, długość nóg, itd.
W PCA staramy się znaleźć takie kombinacje tych cech (składowe główne), które najlepiej wyjaśniają,
jak różnią się dane osoby pod względem tych cech.

Pierwsza składowa (PC1) może być kombinacją cech, która wyjaśnia, np. najwięcej różnic związanych z ogólnym rozmiarem
ciała (np. waga i wzrost).
Druga składowa (PC2) może wyjaśniać różnice, które są bardziej związane z proporcjami ciała (np. dł. nóg, dłoni itp.).
Im większy udział wariancji wyjaśnia składowa, tym ważniejsza jest ta składowa w opisie danych.

Proporcja wyjaśnianej wariancji:

Proporcja wyjaśnionej wariancji dla każdej składowej to stosunek wariancji, którą ta składowa wyjaśnia,
do całkowitej wariancji w danych.
'''