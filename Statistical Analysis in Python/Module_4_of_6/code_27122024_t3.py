# TASK 3 MODULE 4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Insert your code here
df = pd.read_csv('pizza.csv').drop(['id', 'brand'], axis=1)

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

pca = PCA(n_components=6)
pca.fit(df_scaled)

eigenvalues = pca.explained_variance_
prop_var = eigenvalues / np.sum(eigenvalues)

# Plot generating
plt.xlabel("Principal Component")
plt.ylabel("Proportion of Variance Explained")
plt.title("Scree Plot for Proportion of Variance Explained")
plt.grid(True)
plt.plot(np.arange(1, len(prop_var) + 1), prop_var, marker="o")
plt.savefig("fig.png")
