# TASK 3 MODULE 4
# Goal of PCA: Reduce the dimensionality of data while retaining as much of its variability as possible.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Insert your code here
df = pd.read_csv('pizza.csv').drop(['id', 'brand'], axis=1)

# BELOW: The data is standardized so that each feature has a mean of 0 and a standard deviation of 1.
# Standardization is required prior to PCA because this analysis assumes that the data is scaled and features with
#   larger scales would dominate the results.
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# We create a PCA object with the number of principal components.
pca = PCA(n_components=6)
# The fit method fits PCA to standardized data.
pca.fit(df_scaled)
'''
* Components are not directly equivalent to columns, but their number corresponds to the number of columns.
* PRINCIPAL COMPONENTS != COLUMNS
* If you have 7 columns, then the data is in 7-dimensional space, that is: We have 7 columns: mois, prot, fat, ash, sodium, carb, cal.
SO we generate 7 components: PC1, PC2, PC3, ..., PC7.
BUT:
* You can limit the number of components with n_components to a smaller number, if you want to keep only the most important components.
HERE:
* we ignore the least important component PC7 -> which does not mean that we ignore column 7 'cal'. '''


eigenvalues = pca.explained_variance_
prop_var = eigenvalues / np.sum(eigenvalues)
''' -> eigenvalues contains the eigenvalues corresponding to the principal components. Each eigenvalue represents
the amount of variance explained by a given principal component.
    -> prop_var calculates the proportion of variance explained by each principal component.
    This is the ratio of each eigenvalue to the sum of all eigenvalues.
'''

# SCREE PLOT generating /wykres osypiska/
plt.xlabel("Principal Component")
plt.ylabel("Proportion of Variance Explained")
plt.title("Scree Plot for Proportion of Variance Explained")
plt.grid(True)
plt.plot(np.arange(1, len(prop_var) + 1), prop_var, marker="o")
plt.savefig("fig.png")

''' NOTES:
This code performs Principal Component Analysis (PCA) on the data from the pizza.csv file, and then generates
a SCREE PLOT that shows the proportion of variance explained by each principal component.
'''
