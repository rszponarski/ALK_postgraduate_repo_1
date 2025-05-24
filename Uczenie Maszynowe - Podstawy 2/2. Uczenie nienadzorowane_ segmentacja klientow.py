import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import seaborn as sns
import umap
import joblib
from scipy.cluster.hierarchy import linkage, fcluster

# Wczytanie i wstępne przygotowanie danych
np.random.seed(42)

df = pd.read_csv('marketing_campaign.csv')
missing_values = df.isnull().sum()

# Enkodowanie danych
label_encoder = LabelEncoder()
df_categoricals = df.select_dtypes(include=['object'])
df_encoded_categoricals = df_categoricals.apply(label_encoder.fit_transform)

df[df_encoded_categoricals.columns] = df_encoded_categoricals

# Skalowanie danych
df_numerical = df.select_dtypes(include=[np.number])
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numerical)

df[df_numerical.columns] = df_scaled

# K-Means clustering

wcss = []  # Within-Cluster Sum of Squares
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init="k-means++", max_iter=300, n_init=10, random_state=42)
    kmeans.fit(df)
    wcss.append(kmeans.inertia_)

# K-Means clustering II

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Metoda łokcia dla optymalnej liczby klastrów')
plt.xticks(np.arange(1, 11, 1))
plt.xlabel('Liczba klastrów')
plt.ylabel("WCSS")
plt.grid(True)
plt.savefig('elbow_method.png')

kl = KneeLocator(range(1, 11), wcss, curve="convex", direction="decreasing")
optimal_clusters = kl.elbow
# print(optimal_clusters)

# PCA (Principal Component Analysis)

pca = PCA(n_components=2)
df_pca = pd.DataFrame(pca.fit_transform(df), columns=["PC1", "PC2"])

plt.figure(figsize=(7, 6))
plt.scatter(df_pca["PC1"], df_pca["PC2"], alpha=0.5, c="blue")
plt.title('PCA: Redukcja do dwóch wymiarów')
plt.xlabel('PC1')
plt.ylabel("PC2")
plt.savefig('pca.png')

# PCA (Principal Component Analysis) + KMeans

kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df_pca["KMeans_Cluster"] = kmeans.fit_predict(df_pca)

kmeans_score = silhouette_score(df_pca, kmeans.labels_)

# Wizualizacja: KMeans na danych poddanych PCA

sns.scatterplot(data=df_pca, x="PC1", y='PC2', hue="KMeans_Cluster", palette="viridis", s=100, edgecolor="black")
plt.title("K-Means Clustering grup zakupowych po PCA")
plt.savefig("KMeans Clustering after PCA.png")

# UMAP (Uniform and Manifold Projection for Dimension Reduction)

umap_model = umap.UMAP(n_components=2, random_state=42)
df_umap = umap_model.fit_transform(df)
df_umap = pd.DataFrame(df_umap, columns=["UMAP1", "UMAP2"])
print(df_umap)

joblib.dump(umap_model, 'umap_model.pkl')
df_umap.to_csv("df_umap.csv", index=False)

# Hierarchical clustering

link = linkage(df_umap, method='ward')
df_umap["Hierarchical_Cluster"] = fcluster(link, t=optimal_clusters, criterion="maxclust")

hierarchical_score = silhouette_score(df_umap, df_umap["Hierarchical_Cluster"])
print(hierarchical_score)

# Wizualizacja: Hierarchical clustering na danych poddanych UMAP

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_umap, x="UMAP1", y='UMAP2', hue="Hierarchical_Cluster", palette="viridis", s=100, edgecolor="black")
plt.title("Hierarchical Clustering grup zakupowych po UMAP")
plt.savefig("Hierarchical clustering_ after UMAP.png")

# Wizualizacja: Hierarchical clustering na danych poddanych UMAP cd.

df_umap["KMeans_Cluster"] = kmeans.fit_predict(df_umap)

kmeans_score2 = silhouette_score(df_umap, kmeans.labels_)

results = {
    "Model": ["K-Means (PCA)", "K-Means (UMAP)", "Hierarchical (UMAP)"],
    "Score": [kmeans_score,
              kmeans_score2,
              hierarchical_score]
}

results_df = pd.DataFrame(results)
print(results_df)

# Wizualizacja: KMeans na danych poddanych PCA

''' Połączenie klastrów z oryginalnymi danymi'''
df_original_with_clusters = df.copy()
df_original_with_clusters["KMeans_Cluster"] = df_umap["KMeans_Cluster"]

'''Wybrane cechy do analizy'''
features_to_plot = ["WydatkiWino", "Dochod", "WydatkiMieso"]

'''Tworzenie boxplotów'''
for feature in features_to_plot:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="KMeans_Cluster", y=feature, data=df_original_with_clusters, palette="viridis")
    plt.title(f"Rozkład cechy '{feature}' w zależności od klastra")
    plt.xlabel("KMeans Cluster")
    plt.ylabel(feature)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{feature}_boxplot.png")