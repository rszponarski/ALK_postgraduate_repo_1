import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# Wczytanie danych i sprawdzenie braków
wine = pd.read_csv('wine.csv', delimiter=';')

wine_nas = wine.isnull().sum()
#print(wine_nas)

# Histogram
print(wine.columns)  # sprawdzamy parametry, istnieje kolumna 'quality'

plt.figure(figsize=(10, 6))
sns.histplot(wine["quality"], bins=7, color='red', kde=True)
plt.title("Rozkład jakości wina")
plt.xlabel("Jakość")
plt.ylabel("Ilość win")
plt.savefig("wine_histplot.png")

# Macierz korelacji'
plt.figure(figsize=(12, 8))
sns.heatmap(wine.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Rozkład jakości wina")
plt.savefig("wine_macierz_korelacji.png")

"""Modyfikacja kolumny -quality- aby przyjmowała tylko wartości:
 -- 0 (niska jakość));
 -- 1 (wysoka jakość).
 Wartość 1 powinna być przyjmowana od -quality- równej 6 (włącznie) i wyżej, w przeciwnym przypadku - 0. 
 Nadpisujemy oryginalne wartości kolumny "quality". """

print(wine['quality'].unique())
print(wine.shape)
print(wine['quality'])

wine['quality'] = (wine['quality'] >= 6).astype(int)
print(wine.shape)
print(wine['quality'])

"""Podział zbioru na treningowy i testowy"""
X = wine.drop(columns=['quality'])
y = wine['quality']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Zmodyfikuj kod tak, aby proporcje klas w zbiorze treningowym i testowym były takie same
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

train_value_counts = y_train.value_counts(normalize=True)
test_value_counts = y_test.value_counts(normalize=True)

print(train_value_counts)
print(test_value_counts)

"""Standaryzacja cech
- standaryzacja poprzez StandardScaler() (wystandaryzuje cechy danych testowych i treningowych).
Zapis przeskalowanych danych do X_train i X_test."""

scaler = StandardScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

"""MODELE KLASYFIKACJI"""

# Inicjalizacja modeli
models = {
    "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42)
}

# Słowniki do przechowywania wyników
metrics = {"model": [], "accuracy": [], "precision": [], "recall": [], "f1": []}

predictions = {name: None for name in models}

# Funkcja oceny modeli
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[model_name] = y_pred

    accuracy, precision, recall, f1 = evaluate_model(y_test, y_pred)

    metrics['model'].append(model_name)
    metrics['accuracy'].append(accuracy)
    metrics['precision'].append(precision)
    metrics['recall'].append(recall)
    metrics['f1'].append(f1)

results_df = pd.DataFrame(metrics)
print(results_df)

'''Przygotowanie macierzy pomyłek dla wszystkich trzech modeli.'''

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (model_name, y_pred) in enumerate(predictions.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i])
    axes[i].set_title(f"Macierz pomyłek - {model_name}")
    axes[i].set_xlabel('Predykcja')
    axes[i].set_ylabel('Rzeczywistość')

plt.tight_layout()
plt.savefig("wine_macierz_pomylek.png")

"""Krzywa ROC"""

def plot_roc_curve(model, X_test, Y_test, label):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=1)
    auc_score = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"{label}, AUC = {auc_score:.2f}")

plt.figure(figsize=(8, 6))
for model_name, model in models.items():
    plot_roc_curve(model, X_test, y_test, model_name)

plt.plot([0, 1], [0,1], color="red", linestyle='--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Krzywa ROC dla różnych modeli")
plt.legend()
plt.savefig("wine_ROC_krzywa.png")