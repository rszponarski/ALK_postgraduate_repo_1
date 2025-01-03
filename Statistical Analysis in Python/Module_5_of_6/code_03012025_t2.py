# TASK 2 MODULE 5
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
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertias, '-o')
plt.title('Elbow method for seed data')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of squares of distances')
plt.savefig('fig2.png')
