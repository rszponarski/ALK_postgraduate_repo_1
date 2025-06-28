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

texts_train_val, texts_test, y_train_val, y_test = train_test_split(texts, labels, test_size=0.1, random_state=seed)
texts_train, texts_val, y_train, y_val = train_test_split(texts_train_val, y_train_val, test_size=0.2, random_state=seed)

# 3. Tokenizacja

max_len = 128

X_train = tokenizer(
    texts_train,
    max_length=max_len,
    truncation=True,
    padding="max_length",
    return_tensors="tf",
)

X_val = tokenizer(
    texts_val,
    max_length=max_len,
    truncation=True,
    padding="max_length",
    return_tensors="tf",
)

X_test = tokenizer(
    texts_test,
    max_length=max_len,
    truncation=True,
    padding="max_length",
    return_tensors="tf",
)

# 4. Budowa modelu

def build_model():
    bert = TFBertModel.from_pretrained(bert_mini_path, from_pt=True)
    input_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")

    bert_output = bert(input_ids, attention_mask=attention_mask)[1]
    x = tf.keras.layers.Dropout(0.3, seed=seed)(bert_output)
    x = tf.keras.layers.Dense(
        64,
        activation="leaky_relu",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)
    )(x)
    x = tf.keras.layers.Dense(
        6,
        activation="sigmoid",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)
    )(x)

    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=x)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

# --- Zakomentowane trenowanie i zapisywanie modelu ---
# model = build_model()
# history = model.fit(
#     {"input_ids": X_train["input_ids"], "attention_mask": X_train["attention_mask"]},
#     y_train,
#     validation_data=(
#         {"input_ids": X_val["input_ids"], "attention_mask": X_val["attention_mask"]},
#         y_val
#     ),
#     epochs=10,
#     batch_size=32
# )
# model.save("toxic_model.h5", include_optimizer=False)

# --- Załadowanie i ewaluacja zapisanym modelem ---
model = tf.keras.models.load_model(
    "toxic_model.h5",
    custom_objects={"TFBertModel": TFBertModel}
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

loss, acc = model.evaluate(
    {"input_ids": X_test["input_ids"], "attention_mask": X_test["attention_mask"]},
    y_test
)

# =================================================================

# 8. Early stopping

early_stop = tf.keras.callbacks. EarlyStopping(
    monitor="val_accuracy", patience=2, restore_best_weights=True
)

history = model.fit(
    {"input_ids": X_train["input_ids"], "attention_mask": X_train["attention_mask"]},
    y_train,
    validation_data=(
        {"input_ids": X_val["input_ids"], "attention_mask": X_val["attention_mask"]},
        y_val
    ),
    epochs=10,
    batch_size=32
)

loss, acc = model.evaluate(
    {"input_ids": X_test["input_ids"], "attention_mask": X_test["attention_mask"]},
    y_test
)
print (f"\nTest accuracy: {acc:.4f}")

# 9. Predykcje

y_pred_probs = model.predict(
    {"input_ids": X_test["input_ids"], "attention_mask": X_test["attention_mask"]}
)

y_pred = (y_pred_probs > 0.5).astype(int)

# 10. Macierz pomyłek - problem multi-label

cm = multilabel_confusion_matrix(y_test, y_pred)

for i, matrix in enumerate(cm):
    print(f'Klasa {i}:\n{matrix}\n')

'''Zapis macierzy pomyłek do pliku .txt'''

# with open("macierz_pomylek.txt", "w") as f:
#     for i, matrix in enumerate(cm):
#         f.write(f"Klasa {i}:\n{matrix}\n\n")

# 11. Optymalizacja progów odcięcia dla problemu multi-label

y_val_probs = model.predict(
    {"input_ids": X_val["input_ids"], "attention_mask": X_val["attention_mask"]}
)

best_thresholds = []

for i in range(6):
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.01):
        preds = (y_val_probs[:, i] > thresh).astype(int)
        f1 = f1_score(y_val[:, i], preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    best_thresholds.append(best_thresh)

print("Najlepsze progi:", best_thresholds)