import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

#  Wczytanie i podział danych

dataset = pd.read_csv('titanic_train.csv')

'''Oddzielenie kolumny 'Survived' od pozostałych'''
dataset_x = dataset.drop(columns=['Survived'])
dataset_y = dataset['Survived']

train_x, test_x, train_y, test_y = train_test_split(
    dataset_x, dataset_y, train_size=0.8, random_state=42
)

# Wizualizacja
''' Stwórz wykres słupkowy:
Ustaw tytuł wykresu na Survivors za pomocą metody suptitle.
Ustaw podpisy słupków na ['not survived', 'survived'].'''

survivor_counts = train_y.value_counts().sort_index()  # Obliczenie liczby przypadków 0 i 1

plt.figure(figsize=(6, 4))
plt.suptitle('Survivors')
plt.bar(['not survived', 'survived'], survivor_counts)
plt.ylabel('Count')
plt.savefig("survived_or_not.png")

# Usunięcie kolumn
'''  Usuń kolumny PassengerId, Name, Ticket i Cabin z ramki danych dataset_x (znajdują się tam dane, które są mało
istotne dla naszego modelu oraz kolumna Cabin, która w wiekszości zawiera wartości puste) '''

dataset_x = dataset_x.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

# Uzupełnianie brakujących danych

''' Uzupełnienie braków w kolumnie 'Age' wartością średnią'''
mean_age = dataset_x['Age'].mean()
dataset_x['Age'] = dataset_x['Age'].fillna(mean_age)

''' Uzupełnienie braków w kolumnie 'Embarked' najczęściej występującą wartością (most frequent)'''
most_frequent_embarked = dataset_x['Embarked'].mode()[0]
dataset_x['Embarked'] = dataset_x['Embarked'].fillna(most_frequent_embarked)

# Zakodowanie danych

le_sex = LabelEncoder()
le_embarked = LabelEncoder()

dataset_x['Sex'] = le_sex.fit_transform(dataset_x['Sex'])
dataset_x['Embarked'] = le_embarked.fit_transform(dataset_x['Embarked'])
