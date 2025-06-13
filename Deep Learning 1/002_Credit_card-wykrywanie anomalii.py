import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# Wyświetl wszystkie kolumny
# pd.set_option('display.max_columns', None)

# 1. Wstępne przygotowanie danych
df = pd.read_csv("creditcard.csv")

X = df.drop(["Time", "Class"], axis=1).values
y = df["Class"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Podział danych na treningowe i testowe
X_train = X_scaled[y == 0]
X_test= X_scaled

# 3. Budowa enkodera

input_dim = X_train.shape[1]
encoding_dim = 14

input_layer = Input(shape=(input_dim))
encoded = Dense(32, activation="relu")(input_layer)
encoded = BatchNormalization()(encoded)
encoded = Dropout(0.2)(encoded)

encoded = Dense(encoding_dim, activation="relu")(encoded)

# 4. Budowa dekodera

decoded = Dense(32, activation="relu")(encoded)
decoded = BatchNormalization()(decoded)
decoded = Dropout(0.2)(decoded)

decoded = Dense(input_dim, activation="linear")(decoded)

# 5. Budowa autoenkodera

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")

# 6. Trening modelu

autoencoder.fit(X_train, X_train,
                epochs=20,
                batch_size=256,
                shuffle=True,
                validation_split=0.1)

# 7.  Predykcje i błąd rekonstrukcji

X_test_pred = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - X_test_pred, 2), axis=1)

# 8.  Wykrywanie anomalii

threshold = np.percentile(mse[y == 0], 95)
y_pred = (mse > threshold).astype(int)

# 9. Ocena modelu

roc_auc = roc_auc_score(y, mse)
cr = classification_report(y, y_pred)

# 10. Krzywa ROC

'''Obliczenie TPR i FPR'''
fpr, tpr, thresholds = roc_curve(y, mse)

'''Wykres krzywej ROC'''
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Autoencoder (AUC = {roc_auc:.4f})", color='navy')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # linia losowa
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("Roc_curve_credit_card.png")