# TASK 6 MODULE 4

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

# Data loading
df = pd.read_csv("pizza.csv")

# Data pre-processing
df.drop(["id"], axis=1, inplace=True)  # Usuń kolumnę id, jeśli istnieje
features = df.drop(["brand"], axis=1)
brands = df["brand"]

# Standardisation of features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# PCA analysis
pca = PCA(n_components=2)
principal_components = pca.fit_transform(features_scaled)

# Preparation of data for visualisation
pca_df = pd.DataFrame(data = principal_components, columns=['PC1', 'PC2'])
pca_df['Brand'] = brands

# Visualisation of PCA results
plt.figure(figsize=(10, 6))
for i, vector in enumerate(pca.components_.T):
  plt.arrow(
    0, 0, vector[0] * 5, vector[1] * 5, \
    color="r", width=0.005, head_width=0.05)
  plt.text(
    vector[0] * 5.5, vector[1] * 5.5, features.columns[i - 1], \
    color="r")

sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Brand")
plt.title("PCA Scatter Plot Colored by Brand")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Brand")
plt.savefig("fig4.png")