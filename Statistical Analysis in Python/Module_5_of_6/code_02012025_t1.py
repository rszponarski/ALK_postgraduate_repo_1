# TASK 1 MODULE 5

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Data reading
df = pd.read_csv("seeds_dataset.csv")
columns = df.columns

# Insert your code to perform these tasks
x, y = df[columns[5]], df[columns[6]]

# Data standardisation
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Grouping by k-means
kmeans = KMeans(n_clusters=3, random_state=10)
clusters = kmeans.fit_predict(df_scaled)

# Visualisation of results
plt.scatter(x, y, c=clusters, cmap="viridis")

plt.xlabel(columns[5])
plt.ylabel(columns[6])

plt.title("K-means Clustering of Seeds")
plt.savefig("fig1.png")

