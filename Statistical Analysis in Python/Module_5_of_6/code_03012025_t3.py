# TASK 3 MODULE 5
import pandas as pd
# from scipy.stats import pointbiserialr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Data reading
df = pd.read_csv('seeds_dataset.csv')

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

kmeans = KMeans(n_clusters=3, random_state=42)
df['Clusters'] = kmeans.fit_predict(df_scaled)

g = sns.pairplot(df, hue='Clusters', palette='Set2', height=1.6, aspect=1.4, plot_kws={'s':7})
g.add_legend()
plt.savefig('fig3.png')

