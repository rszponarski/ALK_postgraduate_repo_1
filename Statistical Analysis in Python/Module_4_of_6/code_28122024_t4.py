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

# Visualization
plt.scatter(principalComponents[:, 0], principalComponents[:, 1])
plt.title('PCA')
plt.xlabel('First main component')
plt.ylabel('Second main component')
plt.savefig("fig2.png")