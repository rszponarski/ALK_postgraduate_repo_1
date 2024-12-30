# TASK 1 MODULE 4

import pandas as pd

# Data reading
df = pd.read_csv("pizza.csv").drop(["brand", "id"], axis=1)
'''Loads the data and then removes the "brand" and "id" columns because:
* The "brand" column contains text categories that are not needed for correlation calculations.
* The "id" column is a unique identifier that does not affect the analysis of the relationships between variables.
* axis=1 specifies along which axis the column (or row) deletion operation is performed
    axis=1: indicates that the operation applies to columns.
    ["brand", "id"] are the names of the columns to be deleted.
    axis=0: applies to rows.
'''

# The df.corr() function computes the correlation matrix between all columns in a DataFrame.
corr_matrix = df.corr()
# The .corr() function uses Pearson correlation for calculations by default, but you can change the method,
# e.g. to Spearman:     corr_matrix = df.corr(method="spearman")

most_corr = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()
''' Unstack the correlation matrix to a single level:
___unstack(): Converts the correlation matrix from an array format to a single-plane series.
___sort_values(ascending=False): Sorts all pairs of variables by their correlation values in descending order
(highest on top).
___drop_duplicates(): Removes duplicate pairs of variables. The correlation matrix is symmetric
'''

print("Most correlated pairs of variables:")
print(most_corr.head())
print("")

print("Most uncorrelated pairs of variables:")
print(most_corr.tail())

""" NOTES:
What is a correlation matrix?
-> A table that shows the correlation coefficients between all pairs of variables in a data set.
    It is a tool used to examine linear relationships between variables.

Correlation coefficient: A numerical value (usually between -1 and 1) that describes the strength and direction
    of a linear relationship between two variables:
 1: Perfect positive correlation (both variables increase at the same time).
-1: Perfect negative correlation (one variable increases, the other decreases).
 0: No correlation (variables are not linearly related).

*** Why do we use a correlation matrix?
-> It allows us to quickly identify which variables are related.
-> It indicates which variables potentially affect each other.
"""