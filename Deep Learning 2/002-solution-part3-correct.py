import os
import random
import numpy as np
import tensorflow as tf
import keras

# ---Stała wartość seed
seed = 42

# ---Ustawienie seedów dla reprodukowalności
os.environ["PYTHONHASHseed"] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
keras.utils.set_random_seed(seed)

# ---Wymuszenie deterministycznych operacji w TensorFlow
os.environ["TF_DETERMINISTIC_OPS"] = "1"
tf.config.experimental.enable_op_determinism()

# Ścieżka do modelu BERT tiny
bert_mini_path = "prajjwal1/bert-tiny"

import pandas as pd
from transformers import BertTokenizer, TFBertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, f1_score

# 1. Wstępne przygotowanie danych
df = pd.read_csv("toxic_subset.csv")
texts = df["comment_text"].tolist()
labels = df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values

tokenizer = BertTokenizer.from_pretrained(bert_mini_path)

# 2. Podział danych
from sklearn.model_selection import train_test_split
texts_train_val, texts_test, y_train_val, y_test = train_test_split(texts, labels, test_size=0.1, random_state=seed)
texts_train, texts_val, y_train, y_val = train_test_split(texts_train_val, y_train_val, test_size=0.2, random_state=seed)

# 3. Tokenizacja
max_len = 128

X_train = tokenizer(texts_train, max_length=max_len, truncation=True, padding="max_length", return_tensors="tf")
X_val = tokenizer(texts_val, max_length=max_len, truncation=True, padding="max_length", return_tensors="tf")
X_test = tokenizer(texts_test, max_length=max_len, truncation=True, padding="max_length", return_tensors="tf")

# 4. Załadowanie i kompilacja zapisanym modelem
model = tf.keras.models.load_model(
    "toxic_model.h5",
    custom_objects={"TFBertModel": TFBertModel}
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 5. Ewaluacja modelu
loss, acc = model.evaluate(
    {"input_ids": X_test["input_ids"], "attention_mask": X_test["attention_mask"]},
    y_test
)
print(f"\nTest accuracy: {acc:.4f}")

# 6. Predykcje
y_pred_probs = model.predict(
    {"input_ids": X_test["input_ids"], "attention_mask": X_test["attention_mask"]},
    verbose=0
)
y_pred = (y_pred_probs > 0.5).astype(int)

# 7. Macierz pomyłek - zapis do pliku
cm = multilabel_confusion_matrix(y_test, y_pred)

with open("macierz_pomylek.txt", "w") as f:
    for i, matrix in enumerate(cm):
        f.write(f"Klasa {i}:\n{matrix}\n\n")

# 8. Optymalizacja progów F1-score dla każdej klasy
y_val_probs = model.predict(
    {"input_ids": X_val["input_ids"], "attention_mask": X_val["attention_mask"]},
    verbose=0
)

thresholds = np.arange(0.1, 0.9, 0.01)
best_thresholds = []

for i in range(6):
    f1_scores = [(th, f1_score(y_val[:, i], (y_val_probs[:, i] > th).astype(int)))
                 for th in thresholds]
    best_thresh = max(f1_scores, key=lambda x: x[1])[0]
    best_thresholds.append(best_thresh)

print("Najlepsze progi:", best_thresholds)

# 9. Predykcja na zoptymalizowanych progach
y_pred_probs = model.predict(
    {"input_ids": X_test["input_ids"], "attention_mask": X_test["attention_mask"]},
    verbose=0
)

y_pred = np.zeros_like(y_pred_probs, dtype=np.int8)
for i in range(6):
    y_pred[:, i] = (y_pred_probs[:, i] > best_thresholds[i]).astype(np.int8)

# Zmienna wymagana przez test
y_test_pred = y_pred

cm = multilabel_confusion_matrix(y_test, y_test_pred)

for i, matrix in enumerate(cm):
    print(f"Klasa {i}:\n{matrix}\n")

