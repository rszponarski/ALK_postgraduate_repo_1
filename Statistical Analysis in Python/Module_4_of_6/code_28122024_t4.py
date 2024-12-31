# TASK 4 MODULE 4

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Insert your code here
df = pd.read_csv('pizza.csv').drop(['id', 'brand'], axis=1)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

pca = PCA(n_components=3)
principalComponents = pca.fit_transform(scaled_data)
'''PCA: Dimensionality Reduction
-> We create a PCA object with three principal components (n_components=3).
-> The fit_transform function:
    -> fit:         Calculates the principal components and fits the PCA to the data.
    -> transform:   Transforms the input data to new axes (principal components).
principalComponents: Contains the new coordinates of the data in PCA space, where:
principalComponents[:, 0] are the values for PC1 (first principal component),
principalComponents[:, 1] are the values for PC2,
principalComponents[:, 2] are the values for PC3'''

# Visualization
plt.scatter(principalComponents[:, 0], principalComponents[:, 1])
plt.title('PCA')
plt.xlabel('First main component')
plt.ylabel('Second main component')
plt.savefig("fig2.png")

'''
* plt.scatter: Creates a scatter plot in the plane of the first two principal components (PC1 and PC2).
    -> principalComponents[:, 0]: Values of the first principal component (X-axis).
    -> principalComponents[:, 1]: Values of the second principal component (Y-axis).
The visualization shows the data in the space of the first two principal components that best explain the variability
in the data.'''