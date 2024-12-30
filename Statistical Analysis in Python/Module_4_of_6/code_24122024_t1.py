# TASK 1 MODULE 4

import pandas as pd

# Data reading
df = pd.read_csv("pizza.csv").drop(["brand", "id"], axis=1)
'''Loads the data and then removes the "brand" and "id" columns because:
* The "brand" column contains text categories that are not needed for correlation calculations.
* The "id" column is a unique identifier that does not affect the analysis of the relationships between variables.
'''


corr_matrix = df.corr()

most_corr = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()

print("Most correlated pairs of variables:")
print(most_corr.head())
print("")

print("Most uncorrelated pairs of variables:")
print(most_corr.tail())
