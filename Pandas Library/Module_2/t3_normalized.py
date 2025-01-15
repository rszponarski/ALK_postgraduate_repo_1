# PRZYKLAD UZYCIA pd.json_normalize(df[])
# przykład użycia pd.json_normalize() do spłaszczenia zagnieżdżonego obiektu JSON:

# Przykładowe dane
data = [
    {
        "id": 1,
        "name": "Pint",
        "properties": {
            "volume": 500,
            "material": "glass"
        },
        "createDate": "2021-01-01"
    },
    {
        "id": 2,
        "name": "Mug",
        "properties": {
            "volume": 700,
            "material": "ceramic"
        },
        "createDate": "2020-05-10"
    }
]

import pandas as pd

# Spłaszczanie JSON przy użyciu pd.json_normalize()
df = pd.json_normalize(data)
# Wyświetlenie wynikowego DataFrame
print(df, "\n")
"""
Wynikowy DataFrame
   id  name properties.volume properties.material  createDate
0   1  Pint              500              glass  2021-01-01
1   2   Mug              700            ceramic  2020-05-10

JAK WIDAC DOMYSLNYM PREFIXEM JEST KROPKA "."

Wyjaśnienie działania:
    pd.json_normalize(data):
        Spłaszcza każdą zagnieżdżoną strukturę w obiekcie JSON.
        W zagnieżdżonych obiektach (np. properties) dodaje nazwę nadrzędną (properties) jako prefiks.

    Generowane nazwy kolumn:
        Klucze w głównym słowniku JSON stają się nazwami kolumn.
        Klucze w zagnieżdżonym słowniku (np. properties) są łączone z nazwą nadrzędną
        (np. properties.volume, properties.material).
"""

# Opcje zaawansowane
# 1. Dodawanie prefiksu do nazw kolumn: <-- gdy chcemy inny prefix
df_1 = pd.json_normalize(data, sep='_')
print(df_1, "\n")
"""
Wynik (z _ jako separatorem):
   id  name  properties_volume  properties_material  createDate
0   1  Pint               500               glass  2021-01-01
1   2   Mug               700             ceramic  2020-05-10
"""

# 2. Spłaszczanie tylko wybranej części zagnieżdżonego JSON:
# Jeśli chcemy spłaszczyć tylko określony klucz (np. properties):
df_2 = pd.json_normalize(data, record_path=None, meta=['id', 'name', 'createDate'])
print(df_2) # daje taki sam wynik co pd.json_normalize(data) bo properities to jedyna zagniezdzona struktura.
# _____________________________________________________________________________________________________________
# PRZYKLAD z danymi ktore maja wiecej niz jedna zagniezdzona strukture
# Załóżmy, że mamy JSON z dwiema zagnieżdżonymi strukturami:

data_2 = [
    {
        "id": 1,
        "name": "Pint",
        "createDate": "2021-01-01",
        "manufacturer": {
            "name": "BrewCo",
            "location": "New York"
        },
        "properties": [
            {
                "volume": 500,
                "material": "glass"
            }
        ]
    }
]

# Normalizacja obu struktur:
# pd.json_normalize(data) spłaszczy oba zagnieżdżone obiekty (jak manufacturer i properties).

df_3 = pd.json_normalize(data_2)
print(df_3, "\n")

"""
Wynik:
id  name  createDate  manufacturer.name  manufacturer.location  properties.volume  properties.material
0   1  Pint  2021-01-01              BrewCo                  New York               500              glass
"""

# pd.json_normalize(data, record_path='properties') spłaszczy tylko strukturę znajdującą się w properties.
df_properties = pd.json_normalize(data_2, record_path='properties')
print(df_properties)
print(type(df_properties))

"""
Wynik:
volume  material
0      500     glass
"""

"""
Dlaczego pd.json_normalize() jest przydatne?
    --> Automatyczne spłaszczanie: Łatwo przekształca zagnieżdżone JSON-y do postaci tabelarycznej.
"""