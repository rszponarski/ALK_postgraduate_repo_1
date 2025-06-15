import keras
import tensorflow as tf

# 1. Wstępne przygotowanie danych
keras.utils.set_random_seed(43)
tf.config.experimental.enable_op_determinism()

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("sp500.txt")
df = df.sort_values("Date")
data = df[["Close"]].values

train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)