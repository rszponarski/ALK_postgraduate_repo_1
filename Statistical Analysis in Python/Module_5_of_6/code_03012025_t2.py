# TASK 2 MODULE 5
# This code implements the Elbow Method to determine the optimal number of clusters (k) for the K-means algorithm
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Data reading
df = pd.read_csv('seeds_dataset.csv')

# Data standardisation
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Creating an elbow method chart
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    inertias.append(kmeans.inertia_)  # See note below
'''Inertia is the sum of the squared distances of data points from their nearest cluster center.
It is a measure of the "density" of clusters.
Smaller values of inertia mean that the points in the clusters are more concentrated around their center.'''

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertias, '-o')  # '-o': Lines connected by dots.
plt.title('Elbow method for seed data')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of squares of distances')
plt.savefig('fig2.png')
