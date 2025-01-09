# TASK 1 MODULE 5

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
# implementation of the k-means algorithm for grouping data into clusters.
import matplotlib.pyplot as plt

# Data reading
df = pd.read_csv("seeds_dataset.csv")
columns = df.columns

# Selecting columns to visualize
x, y = df[columns[5]], df[columns[6]]
# Asymmetry Coefficient,Length of kernel groove
# x and y are the selected data columns corresponding to columns 6 and 7 (zero-based indexing).
# These columns are used later to create the chart.

# Data standardisation
scaler = StandardScaler()
# StandardScaler transforms the data so that each feature has a mean of 0 and a standard deviation of 1.
df_scaled = scaler.fit_transform(df)
# Standardization improves the performance of clustering algorithms such as k-means,
#   which are sensitive to differences in data scales.

# Grouping by k-means
kmeans = KMeans(n_clusters=3, random_state=10)
clusters = kmeans.fit_predict(df_scaled)

# Visualisation of results
plt.scatter(x, y, c=clusters, cmap="viridis")

plt.xlabel(columns[5])
plt.ylabel(columns[6])

plt.title("K-means Clustering of Seeds")
plt.savefig("fig1.png")

