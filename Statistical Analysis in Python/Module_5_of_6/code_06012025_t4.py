# TASK 4 MODULE 5
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Data reading
df = pd.read_csv("seeds_dataset.csv")

X = df.drop(["Type of seed"], axis=1)
y_true = df["Type of seed"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
y_pred = kmeans.fit_predict(X_scaled)

ari = adjusted_rand_score(y_true, y_pred)
print(f'Adjusted Rand Index: {ari}')
