import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

'''Wczytanie i wstępny podział danych'''
df = pd.read_csv('iris.csv')
print(df.columns)

# Podział na cechy (X) i etykiety (y)
X = df.drop(columns=['species'])
y = df['species']

# print("X shape:", X.shape) # Opcjonalnie sprawdzenie kształtów
# print("y shape:", y.shape)

'''Skalowanie danych '''

scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)  # Skalowanie danych

# scaled_X to teraz NumPy array. Można przekonwertować go z powrotem do DataFrame z tymi samymi nazwami kolumn co w X:
# scaled_X = pd.DataFrame(scaled_X, columns=X.columns)

'''Podział na dane treningowe i testowe'''

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=0
)

'''Budowa i trening modelu KNN'''

model = KNeighborsClassifier(n_neighbors=3)  # Tworzenie modelu KNN z 3 sąsiadami
model.fit(X_train, y_train)  # Trenowanie modelu na zbiorze treningowym

'''Wytrenowanie modelu '''
# Dokonaj predykcji danych na swoim modelu KNeighborsClassifier, przypisz je do zmiennej y_pred.

y_pred = model.predict(X_test)  # Predykcja na danych testowych

'''Zakodowanie danych '''

encoder = LabelEncoder()
encoded_iris = encoder.fit_transform(y)  # Kodowanie etykiet gatunków irysów

'''Wykorzystanie modelu SVC '''


