# TASK 5 MODULE 4

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

# Data loading
df = pd.read_csv("pizza.csv")

# Data pre-processing
df = df.drop(['id'], axis=1)
features = df.drop(['brand'], axis=1)
brands = df['brand']

# Standardization of features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# PCA analysis
pca = PCA(n_components=2)
principal_components = pca.fit_transform(features_scaled)

# Preparation of data for visualization
pca_df = pd.DataFrame(data = principal_components, columns=['PC1', 'PC2'])
pca_df['Brand'] = brands

# Visualization of PCA results
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Brand")
plt.title("PCA Scatter Plot Colored by Brand")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Brand")
plt.savefig("fig3.png")