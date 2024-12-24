# TASK 1 MODULE 4

import pandas as pd

# Data reading
df = pd.read_csv("pizza.csv").drop(["brand", "id"], axis=1)

corr_matrix = df.corr()

most_corr = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()

print("Most correlated pairs of variables:")
print(most_corr.head())
print("")

print("Most uncorrelated pairs of variables:")
print(most_corr.tail())
