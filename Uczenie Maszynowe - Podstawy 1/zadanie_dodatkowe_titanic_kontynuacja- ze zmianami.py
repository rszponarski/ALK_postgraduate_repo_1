import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Wczytanie i podział danych
dataset = pd.read_csv('titanic_train.csv')

# Oddzielenie kolumny 'Survived' od pozostałych
dataset_x = dataset.drop(columns=['Survived'])
dataset_y = dataset['Survived']

# Usunięcie nieistotnych kolumn
dataset_x = dataset_x.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

# Uzupełnianie braków
mean_age = dataset_x['Age'].mean()
dataset_x['Age'] = dataset_x['Age'].fillna(mean_age)

most_frequent_embarked = dataset_x['Embarked'].mode()[0]
dataset_x['Embarked'] = dataset_x['Embarked'].fillna(most_frequent_embarked)

# Kodowanie danych
le_sex = LabelEncoder()
le_embarked = LabelEncoder()
dataset_x['Sex'] = le_sex.fit_transform(dataset_x['Sex'])
dataset_x['Embarked'] = le_embarked.fit_transform(dataset_x['Embarked'])

# Podział danych z random_state=42
train_x, test_x, train_y, test_y = train_test_split(
    dataset_x, dataset_y, train_size=0.8, random_state=42
)

# Standaryzacja
scaler = StandardScaler()
train_x_scaled = pd.DataFrame(scaler.fit_transform(train_x), index=train_x.index, columns=train_x.columns)
test_x_scaled = pd.DataFrame(scaler.transform(test_x), index=test_x.index, columns=test_x.columns)

# Zastąpienie oryginalnych ramek przeskalowanymi
train_x = train_x_scaled
test_x = test_x_scaled

# Wizualizacja
survivor_counts = train_y.value_counts().sort_index()
plt.figure(figsize=(6, 4))
plt.suptitle('Survivors')
plt.bar(['not survived', 'survived'], survivor_counts)
plt.ylabel('Count')
plt.savefig("survived_or_not_2.png")

'''Stwórz model SGDClassifier z modułu sklearn.linear_model z parametrami konstruktora loss="hinge", penalty="l2".
Przypisz go do zmiennej model_sgd. Następnie wytrenuj go na zbiorach train_x i train_y. '''

# Tworzenie modelu SGDClassifier
model_sgd = SGDClassifier(loss="hinge", penalty="l2")

# Trenowanie modelu
model_sgd.fit(train_x, train_y)

''' Stwórz model svm.SVC z modułu sklearn z domyślnymi parametrami konstruktora (nie podawaj nic).
Przypisz go do zmiennej model_svc. Następnie wytrenuj model na zbiorach train_x i train_y. '''

# Tworzenie modelu SVC z domyślnymi parametrami
model_svc = SVC()

# Trenowanie modelu
model_svc.fit(train_x, train_y)

''' Stwórz model DecisionTreeClassifier z modułu sklearn.tree z domyślnymi parametrami konstruktora (nie podawaj nic).
Przypisz go do zmiennej model_tree. Następnie wytrenuj go na zbiorach train_x i train_y. '''

# Tworzenie modelu drzewa decyzyjnego z domyślnymi parametrami
model_tree = DecisionTreeClassifier()

# Trenowanie modelu
model_tree.fit(train_x, train_y)

'''Dokonaj predykcji za pomocą wytrenowanych modeli: model_sgd, model_svc i model_tree.
Następnie porównaj wyniki tych modeli — możesz do tego wykorzystać moduł metrics z biblioteki sklearn.
Dla każdego modelu stwórz obiekt przygotowanej klasy Metric, zawierający metryki: accuracy, f1 score,
mean absolute error oraz współczynnik determinacji (R² Score). Zapisz te obiekty w słowniku metrics,
używając nazw modeli jako kluczy. '''

from sklearn.linear_model import SGDClassifier
from sklearn import svm, tree
from sklearn import metrics as sk_metrics
from dataclasses import dataclass
from typing import Union, Any

# Definicja klasy Metric z dataclass
@dataclass
class Metric:
    model: Union[SGDClassifier, svm.SVC, tree.DecisionTreeClassifier]
    accuracy: Any
    f1: Any
    mean_abs_error: Any
    r2: Any

# Predykcje
pred_sgd = model_sgd.predict(test_x)
pred_svc = model_svc.predict(test_x)
pred_tree = model_tree.predict(test_x)

# Słownik na metryki
metrics = {
    "SGDClassifier": Metric(
        model=model_sgd,
        accuracy=sk_metrics.accuracy_score(test_y, pred_sgd),
        f1=sk_metrics.f1_score(test_y, pred_sgd),
        mean_abs_error=sk_metrics.mean_absolute_error(test_y, pred_sgd),
        r2=sk_metrics.r2_score(test_y, pred_sgd)
    ),
    "SVC": Metric(
        model=model_svc,
        accuracy=sk_metrics.accuracy_score(test_y, pred_svc),
        f1=sk_metrics.f1_score(test_y, pred_svc),
        mean_abs_error=sk_metrics.mean_absolute_error(test_y, pred_svc),
        r2=sk_metrics.r2_score(test_y, pred_svc)
    ),
    "DecisionTreeClassifier": Metric(
        model=model_tree,
        accuracy=sk_metrics.accuracy_score(test_y, pred_tree),
        f1=sk_metrics.f1_score(test_y, pred_tree),
        mean_abs_error=sk_metrics.mean_absolute_error(test_y, pred_tree),
        r2=sk_metrics.r2_score(test_y, pred_tree)
    )
}